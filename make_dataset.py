import netCDF4 as nc
import h5py
from fire import Fire
import os
import numpy as np
from global_land_mask import globe
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from mpi4py import MPI
from collections import defaultdict

"""
Create the ML dataset in parallel.
"""

def determine_landfall(track_df):
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
                                t=i
                                break
                was_land = is_land

        if t is None:
            return None, None
        # now that we know the landfall time (roughly), determine the exact coordinates
        start, end = track_coords[t-1], track_coords[t]
        for lamb in np.linspace(0,1,100):
                point = (1-lamb) * start + lamb * end
                if globe.is_land(*point):
                        return (1-lamb)*hours[t-1] + lamb * hours[t], point
        return None, None


def extract_storm_data(dirname, coastal_dists, hours_before, hours_after, 
                       cutoff_coastal_dist, max_depth, min_depth,
                      r, downsample_factor):

    """Convert the underlying forcing and best-track output into something usable for ML.
    """

    df = pd.read_csv(dirname+"/best_track.csv")

    # time is in hours since simulation start
    time, landfall_coord = determine_landfall(df)
    if time is None:
        return

    # Check for custom coastal distances
    coastal_file = dirname+"/coastal_dist.hdf5"
    if os.path.exists(coastal_file):
        with h5py.File(coastal_file) as coastal_ds:
            coastal_dists = coastal_ds["dist"][:]
    
    with nc.Dataset(dirname+"/fort.73.nc", "r") as pressure_ds:

        times = pressure_ds["time"][:] / 3600.
        time_inds = np.where((times >= (time - hours_before)) & (times <= (time + hours_after)))[0]
        if not len(time_inds):
            return

        depth = pressure_ds["depth"][:]

        x = pressure_ds["x"][:]
        y = pressure_ds["y"][:]
        
        coords = np.deg2rad(np.column_stack([y, x]))
        landfall_dists = haversine_distances(coords, np.deg2rad(landfall_coord).reshape((1,2))).flatten()
        include = np.arange(len(x)) % downsample_factor == 0
        mask = ((coastal_dists < cutoff_coastal_dist) & (landfall_dists < r) &
                (depth < max_depth) & (depth > min_depth) & include)

        inds = np.where(mask)[0][::10]
        if not len(inds):
            return
        
        pressure = pressure_ds["pressure"][time_inds][:, inds].T
    
    with nc.Dataset(dirname+"/fort.74.nc", "r") as wind_ds:
        windx = wind_ds["windx"][time_inds][:, inds].T
        windy = wind_ds["windy"][time_inds][:, inds].T

    with nc.Dataset(dirname+"/maxele.63.nc", "r") as maxele_ds:
        maxele = maxele_ds["zeta_max"][inds]

    res = {"x": x[inds], "y": y[inds], "coastal_dist": coastal_dists[inds],
                "landfall_dist": landfall_dists[inds], "depth": depth[inds],
                "windx": windx, "windy": windy, "pressure": pressure,
                "landfall_location": landfall_coord.reshape((1,2)), "maxele": maxele,
            "inds": inds}
        
    dt = times[1] - times[0]
    expected_size = (hours_before+hours_after) / dt
    #print(expected
    if len(time_inds) < expected_size:
        for k in ["pressure", "windx", "windy"]:
            new_arr = np.full((len(inds), int(expected_size)), np.nan)
            new_arr[:, :len(time_inds)] = res[k]
            res[k] = new_arr
    elif len(time_inds) > expected_size:
        for k in ["pressure", "windx", "windy"]:
            res[k] = res[k][:, :int(expected_size)]
        
    return res
        
default_kwargs = {
    "hours_before":6,
    "hours_after":6,
    "cutoff_coastal_dist":30,
    "max_depth":2,
    "min_depth":-4,
    "r":150,
    "downsample_factor":10
}

def main(name, datadir="data", stormsdir="storms", **kwargs):
    """Process each storm in the given directory in parallel and combine the results into a single dataset.
    """

    params = default_kwargs.copy()
    params.update(kwargs)
    arrs = defaultdict(list)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0    

    if rank == root:
        dirs = []
        stormsdir = datadir+"/"+stormsdir
        for d in os.listdir(stormsdir):
            d = stormsdir+"/"+d
            if os.path.isdir(d) and "." not in d:
                dirs.append(d)

        with h5py.File(datadir+"/coastal_dist.hdf5", "r") as coastal_dist_ds:
            coastal_dists = coastal_dist_ds["dist"][:]
    else:
        dirs = None

    dirs = comm.bcast(dirs, root=root)
    nnodes = None if rank != root else len(coastal_dists)
    nnodes = comm.bcast(nnodes, root=root)
    if rank != root:
        coastal_dists = np.empty(nnodes, dtype=float)

    comm.Bcast([coastal_dists, MPI.DOUBLE], root=root)
    
    for i in range(rank, len(dirs), size):
            info = extract_storm_data(dirs[i], coastal_dists, **params)
            if info is None:
                    print(f"Storm {i} missing data.")
                    continue
            info["storm"] = np.full(len(info["inds"]), i, dtype=int)
            for k, v in info.items(): arrs[k].append(v)
        

    local_data = {}
    for k, v in arrs.items():
        if isinstance(v[0], np.ndarray):
            local_data[k] = np.concatenate(v)
        else:
            #list of scalars
            local_data[k] = np.array(v)

    keys = sorted(list(local_data.keys()))

    data = {}
    for k in keys:
        counts = np.array(comm.gather(len(local_data[k]), root))
        if rank == root:
            local_shape = local_data[k].shape
            buf_shape = (sum(counts),) + local_shape[1:] if len(local_shape) > 1 else (sum(counts),)
            recvbuf = np.empty(buf_shape, dtype=local_data[k].dtype)
            flatcounts = recvbuf.size // buf_shape[0] * counts
            recvbuf = recvbuf.flatten()
            print(k)
        else:
            flatcounts = recvbuf = None
        comm.Gatherv(sendbuf=local_data[k].flatten(), recvbuf=(recvbuf, flatcounts), root=root)
        if rank == root:
            data[k] = recvbuf.reshape(buf_shape)
            print(f"Processed {k}", data[k].shape)

    if rank != root: return

    with h5py.File(f"{datadir}/datasets/{name}.hdf5", "w") as outds:
        for k, v in data.items():
            outds[k] = v
        
        outds["storm_names"] = np.array([os.path.basename(d) for d in dirs], dtype="S")
        for param_name, param_value in params.items():
            outds.attrs[param_name] = param_value
        
if __name__ == "__main__":
    Fire(main)