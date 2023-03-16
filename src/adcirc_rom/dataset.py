"""
Datasets

Setup and create datasets.
"""
import os
import pdb

import h5py
import netCDF4 as nc
import numpy as np
import pandas as pd
from fire import Fire
from global_land_mask import globe
from sklearn.metrics.pairwise import haversine_distances

try:
    from mpi4py import MPI

    have_mpi = True
except ImportError:
    print(
        "Warning - could not import MPI - dataset creation will be significantly slower!"
    )
    have_mpi = False

import gc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from alive_progress import alive_bar

from adcirc_rom.features import GridEncoder, init_shared_features

earth_radius = 6731


def determine_landfall(track_df):
    """
    Determine Landfal

    Compute the landfall time and location of a storm based on its track data.
    If the track data includes the "Hours" column, it is used
    to determine the time of landfall. Otherwise, the function assumes that the
    time of landfall is evenly spaced between the points in the track data. The
    function first checks whether the storm made landfall in the Gulf of
    Mexico, between latitudes 24 and 31 degrees and longitudes -98 and -88
    degrees. If it did, the function uses linear interpolation to determine the
    exact time and location of landfall.

    Parameters
    ----------
    track_df : pandas DataFrame
        A DataFrame containing the storm's track data, including columns
        "Storm Latitude", "Storm Longitude", and optionally "Hours".

    Returns
    -------
    tuple or None
        A tuple containing the landfall time (in hours) and location (latitude,
        longitude) of the storm, or None if the storm did not make landfall.

    Notes
    -----

    """
    track_coords = track_df[["Storm Latitude", "Storm Longitude"]].values
    if "Hours" in track_df.columns:
        hours = track_df["Hours"]
    else:
        hours = np.arange(len(track_coords))

    was_land = False
    t = None
    for i, latlon in enumerate(track_coords):
        is_land = globe.is_land(*latlon)
        lat, lon = latlon
        if not was_land and is_land:
            if 24 < lat < 31 and -98 < lon < -88:
                t = i
                break
        was_land = is_land

    if t is None:
        return None, None
    # now that we know the landfall time (roughly), determine the exact coordinates
    start, end = track_coords[t - 1], track_coords[t]
    for lamb in np.linspace(0, 1, 100):
        point = (1 - lamb) * start + lamb * end
        if globe.is_land(*point):
            return (1 - lamb) * hours[t - 1] + lamb * hours[t], point
    return None, None


class Dataset:
    """A class containing various dataset tools"""

    def _extract_storm_data(
        self,
        dirname,
        hours_before,
        hours_after,
        cutoff_coastal_dist,
        max_depth,
        min_depth,
        r,
        downsample_factor,
        bounds,
    ):
        """Convert the underlying forcing and best-track output into something usable for ML."""

        # initialize return array
        res = {}

        trackfile = dirname + "/best_track.csv"
        if not os.path.exists(trackfile):
            print("No best-track file found - not performing temporal localization!")
            print(f"Spatial localization will be performed with the bounds {bounds}")
            time = 0
            hours_after = np.inf
            landfall_coord = None
        else:
            df = pd.read_csv(trackfile)
            # time is in hours since simulation start
            time, landfall_coord = determine_landfall(df)
            if time is None:
                return

        # precomputed mesh variables like distance to coast and bathymetry stats
        local_mesh_vars = self._get_mesh_vars(dirname)
        for k in self.mesh_vars:
            if k not in local_mesh_vars:
                local_mesh_vars[k] = self.mesh_vars[k]

        with nc.Dataset(dirname + "/fort.73.nc", "r") as pressure_ds:
            times = pressure_ds["time"][:] / 3600.0
            time_inds = np.where(
                (times >= (time - hours_before)) & (times <= (time + hours_after))
            )[0]
            if not len(time_inds):
                return

            depth = pressure_ds["depth"][:]

            x = pressure_ds["x"][:]
            y = pressure_ds["y"][:]
            mask = (
                (local_mesh_vars["coastal_dist"] < cutoff_coastal_dist)
                & (depth < max_depth)
                & (depth > min_depth)
                & (x <= bounds[3])
                & (x >= bounds[2])
                & (y <= bounds[1])
                & (y >= bounds[0])
            )

            if landfall_coord is not None:
                coords = np.deg2rad(np.column_stack([y, x]))
                landfall_dists = (
                    haversine_distances(
                        coords, np.deg2rad(landfall_coord).reshape((1, 2))
                    ).flatten()
                    * earth_radius
                )
                mask &= landfall_dists < r

            inds = np.where(mask)[0][::downsample_factor]
            if not len(inds):
                return

            pressure = pressure_ds["pressure"][time_inds]

        with nc.Dataset(dirname + "/fort.74.nc", "r") as wind_ds:
            windx = wind_ds["windx"][time_inds]
            windy = wind_ds["windy"][time_inds]

        magnitude = (windx**2 + windy**2) ** 0.5
        forcing_vars = {
            "pressure": pressure,
            "magnitude": magnitude,
            "windx": windx,
            "windy": windy,
        }

        if os.path.exists(dirname + "/fort.93.nc"):
            with nc.Dataset(dirname + "/fort.93.nc") as ice_ds:
                forcing_vars["iceaf"] = ice_ds["iceaf"][time_inds]

        encoder = GridEncoder(x, y, resolution=0.01, bounds=bounds)

        for name, arr in forcing_vars.items():
            for pref, func in {"min": np.min, "max": np.max, "mean": np.mean}.items():
                stat = func(arr, axis=0)
                stat_name = pref + "_" + name
                res[stat_name] = stat[inds]
                computed_vars = encoder.encode(
                    stat,
                    scales=[10, 20, 40],
                    outx=x[inds],
                    outy=y[inds],
                    name=stat_name,
                )
                res.update(computed_vars)

        with nc.Dataset(dirname + "/maxele.63.nc", "r") as maxele_ds:
            maxele = maxele_ds["zeta_max"][inds]

        res.update(
            {
                "x": x[inds],
                "y": y[inds],
                "depth": depth[inds],
                "maxele": maxele,
                "inds": inds,
            }
        )

        if landfall_coord is not None:
            res.update(
                {
                    "landfall_dist": landfall_dists[inds],
                    "landfall_location": landfall_coord.reshape((1, 2)),
                }
            )

        for k, arr in local_mesh_vars.items():
            res[k] = arr[inds]

        return res

    def _get_mesh_vars(self, dirname):
        """Load variables that depend on the mesh from a given directory"""

        res = {}
        coastal_file = dirname + "/coastal_dist.hdf5"
        if os.path.exists(coastal_file):
            with h5py.File(coastal_file) as coastal_ds:
                res["coastal_dist"] = coastal_ds["dist"][:]

        bathy_file = dirname + "/bathy_stats.hdf5"
        if os.path.exists(bathy_file):
            with h5py.File(bathy_file) as bathy_ds:
                for k in bathy_ds.keys():
                    res[k] = bathy_ds[k][:]

        # check consistency
        num_nodes = None
        for k, arr in res.items():
            if num_nodes is None:
                num_nodes = len(arr)
            if len(arr) != num_nodes:
                raise ValueError(
                    f"Inconsistent lengths - arr {k} has length {len(arr)} != {num_nodes}"
                )

        return res

    def _init_shared_arrs(self, datadir):
        """Initialize the shared memory"""

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        root = 0

        itemsize = MPI.DOUBLE.Get_size()
        if rank == root:
            mesh_vars = self._get_mesh_vars(datadir)
            var_names = list(mesh_vars.keys())
            num_nodes = len(mesh_vars[var_names[0]])
        else:
            num_nodes = 0
            mesh_vars = {}
            var_names = None

        num_nodes = comm.bcast(num_nodes, root=root)
        var_names = comm.bcast(var_names, root=root)
        self.mesh_vars = {}
        for v in var_names:
            win = MPI.Win.Allocate_shared(num_nodes * itemsize, itemsize, comm=comm)
            buf, itemsize = win.Shared_query(0)
            self.mesh_vars[v] = arr = np.ndarray(
                buffer=buf, dtype="d", shape=(num_nodes,)
            )
            if rank == 0:
                arr[:] = mesh_vars[v]
        comm.Barrier()

    def _mpi_get_data(self, name, datadir, stormsdir, params):
        """Create a dataset in parallel"""

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        if rank == root:
            dirs = self._get_storm_dirs(datadir, stormsdir)
        else:
            dirs = None

        dirs = comm.bcast(dirs, root=root)
        self._init_shared_arrs(datadir)

        local_data = self._get_data(dirs, params, inds=range(rank, len(dirs), size))
        keys = sorted(list(local_data.keys()))

        data = {}
        for k in keys:
            counts = np.array(comm.gather(len(local_data[k]), root))
            if rank == root:
                local_shape = local_data[k].shape
                buf_shape = (
                    (sum(counts),) + local_shape[1:]
                    if len(local_shape) > 1
                    else (sum(counts),)
                )
                recvbuf = np.empty(buf_shape, dtype=local_data[k].dtype)
                flatcounts = recvbuf.size // buf_shape[0] * counts
                recvbuf = recvbuf.flatten()
                # print(k)
            else:
                flatcounts = recvbuf = None
            comm.Gatherv(
                sendbuf=local_data[k].flatten(),
                recvbuf=(recvbuf, flatcounts),
                root=root,
            )
            if rank == root:
                data[k] = recvbuf.reshape(buf_shape)
                print(f"Processed {k}", data[k].shape)
            del local_data[k]
            gc.collect()

        if rank != root:
            return
        self._save_dataset(name, datadir, data, dirs, params)

    def _save_dataset(self, name, datadir, data, dirs, params):
        """Write a newly assembled dataset to a file"""

        with h5py.File(f"{datadir}/datasets/{name}.hdf5", "w") as outds:
            for k, v in data.items():
                outds[k] = v
            print("Wrote all data items")
            outds["storm_names"] = np.array(
                [os.path.basename(d) for d in dirs], dtype="S"
            )
            for param_name, param_value in params.items():
                outds.attrs[param_name] = param_value

    def _get_storm_dirs(self, datadir, stormsdir):
        dirs = []
        dirname = datadir + "/" + stormsdir
        for d in sorted(os.listdir(dirname)):
            d = dirname + "/" + d
            if os.path.isdir(d) and "." not in d:
                # ensemble outputs subdirectory
                if os.path.exists(d + "/outputs"):
                    d = d + "/outputs"
                dirs.append(d)

        return dirs

    def _get_data(self, dirs, params, inds=None):
        """
        Aggregate feature data for a set of storms, possibly indexed by inds.

        Parameters:
        -----------
        dirs : List[str]
            A list of directories containing storm data.
        params : Dict[str, Union[str, int, float]]
            A dictionary of parameters to pass to _extract_storm_data() function.
        inds : Optional[List[int]], default None
            A list of indices to loop over from the given directories. Default is None, which loops over all directories.

        Returns:
        --------
        Dict[str, Union[np.ndarray, List[Union[str, int, float]]]]
            A dictionary containing feature data for a set of storms.
        """

        arrs = defaultdict(list)
        if inds is None:
            inds = range(len(dirs))
        with alive_bar(
            len(inds),
            length=10,
            title=f"Processing {len(inds)} storms...",
            bar="circles",
            dual_line=True,
            force_tty=True,
            monitor=True,
        ) as bar:
            for i in inds:
                bar.text = f"-> Extracting data for storm {i}"
                info = self._extract_storm_data(dirs[i], **params)
                if info is None:
                    print(f"Storm {i} missing data.")
                    continue
                info["storm"] = np.full(len(info["inds"]), i, dtype=int)
                for k, v in info.items():
                    arrs[k].append(v)
                gc.collect()
                bar()

        local_data = {}
        for k, v in arrs.items():
            if isinstance(v[0], np.ndarray):
                local_data[k] = np.concatenate(v)
            else:
                # list of scalars
                local_data[k] = np.array(v)

        del arrs
        gc.collect()
        return local_data

    def create(
        self,
        name,
        datadir="data",
        stormsdir="storms",
        parallel=False,
        hours_before=6,
        hours_after=6,
        cutoff_coastal_dist=30,
        max_depth=2,
        min_depth=-4,
        r=150,
        downsample_factor=100,
        bounds=(24, 32, -98, -88),
    ):
        """
        Create

        Creates a new dataset by loading and processing storm data stored in
        the directory specified by `datadir`. The processed data is then stored
        in a format that is suitable for use with the model.py library and
        entrypoints to train and validate models.

        Parameters
        ----------
        name : str
            The name of the dataset to create.
        datadir : str, optional
            The path to the directory where the dataset will be stored.
            Default is "data".
        stormsdir : str, optional
            The name of the directory that contains the storm data.
            Default is "storms".
        parallel: bool, optional
            Whether to run in parallel using MPI (if available).
            Default is False.
        **kwargs
            Additional keyword arguments for configuring the dataset creation
            process.

        Returns
        -------
        None

        Notes
        -----
        With the default settings, dataset creation in a serial environment
        (i.e. Designsafe jupyter) will take a few hours due to the lack of MPI
        support and the size of the data to be processed. The dataset
        generation script supports parallization with MPI - and is
        significantly faster when run on HPC resources such as TACC.
        """

        params = {
            "hours_before": hours_before,
            "hours_after": hours_after,
            "cutoff_coastal_dist": cutoff_coastal_dist,
            "max_depth": max_depth,
            "min_depth": min_depth,
            "r": r,
            "downsample_factor": downsample_factor,
            "bounds": bounds,
        }

        if have_mpi and parallel:
            self._mpi_get_data(name, datadir, stormsdir, params)
            return

        print("Getting storm directories...")
        dirs = self._get_storm_dirs(datadir, stormsdir)
        print("Getting mesh variables...")
        self.mesh_vars = self._get_mesh_vars(datadir)
        print("Computing storm features...")
        data = self._get_data(dirs, params)

        print("Saving Dataset...")
        self._save_dataset(name, datadir, data, dirs, params)

    def setup(
        self,
        datadir="data",
        projectdir=os.path.expandvars("$HOME/NHERI-Published/PRJ-2968"),
    ):
        """
        Setup

        Setup the needed folder structure for analysis to work with the
        subdirectories:
          - datasets: The datasets directory is used for storing
          machine-learning ready datasets.
          - storms:The storms directory will contain the raw ADCIRC input
          (note when run within the DesignSafe environment this directory will
          be prepopulated with a dataset of 446 synthetic ADCIRC simulations).
          - models: Finally, the models dataset is used for storing saved ML
          models and predictions.

        Parameters
        ----------
        datadir : str, optional
            The data directory to be created, by default "data".
        projectdir : str, optional
            The project directory, by default
            os.path.expandvars("$HOME/NHERI-Published/PRJ-2968").

        Returns
        -------
        None

        Notes
        -----
        Should be run once before doing work (running create, train. etc.)
        """

        os.makedirs(datadir, exist_ok=True)
        os.makedirs(datadir + "/storms", exist_ok=True)
        os.makedirs(datadir + "/datasets", exist_ok=True)
        os.makedirs(datadir + "/models", exist_ok=True)

        fema_storms = projectdir + "/storms"
        for d in os.listdir(fema_storms):
            dirname = fema_storms + "/" + d
            if os.path.isdir(dirname) and d.startswith("s"):
                newdir = datadir + "/storms/" + d
                os.makedirs(newdir, exist_ok=True)
                os.system(f"ln -sf {dirname}/*nc {newdir}")

        # fix best track
        df = pd.read_csv(projectdir + "/best_tracks.csv", skiprows=[1, 2])
        for idx, group in df.groupby("Storm ID"):
            group = group[
                [
                    "Central Pressure",
                    "Forward Speed",
                    "Heading",
                    "Holland B1",
                    "Radius Max Winds",
                    "Radius Pressure 1",
                    "Storm Latitude",
                    "Storm Longitude",
                ]
            ]
            group.to_csv(
                datadir + f"/storms/s{int(idx):03}/best_track.csv", index=False
            )

        init_shared_features(input_dir=datadir + "/storms/s001", output_dir=datadir)


if __name__ == "__main__":
    Fire(Dataset)
