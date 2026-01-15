"""File for vision-oriented feature preprocessing."""

from feature_functions import HollandWinds
import numpy as np
import pandas as pd
import h5py
import glob
from sklearn.neighbors import BallTree
import pickle
import os
from global_land_mask import globe
from constants import earth_radius, BASINS

# TODO - add functions for visualization of gridded features
# TODO - analyze time of zeta max (relative to time of landfall)
# TODO - get bathymetry gradients
TRACK_DT = 3

class StormData:
    
    def __init__(self, track, landfall, zeta, zeta_time):
        self.track = track
        self.landfall_row = landfall
        self.zeta = zeta
        self.zeta_time = zeta_time
        self.basin_str = BASINS[int(landfall['basin'])]
        self._precise_landfall()
    
    def _precise_landfall(self):
        """Determine exact point and time where storm crosses land."""
        
        landfall_ind = int(min(4*8, self.landfall_row['tstep']))
        prev_ind = landfall_ind - 1
        track_lats, track_lons = self.track['lat'].values, self.track['lon'].values
        lats = np.linspace(track_lats[prev_ind], track_lats[landfall_ind], 11)
        lons = np.linspace(track_lons[prev_ind], track_lons[landfall_ind], 11)
        lons[lons>180] -= 360
        is_land = globe.is_land(lats, lons)
        for i, land in enumerate(is_land):
            if land:
                self.landfall_lat = lats[i]
                self.landfall_lon = lons[i]
                self.landfall_time = landfall_ind - 1 + float(i)/(len(lats)-1)
                return
        #print("no landfall according to global_land_mask")
        self.landfall_lat = self.landfall_row['lat']
        self.landfall_lon = self.landfall_row['lon']
        self.landfall_time = landfall_ind

def load_storms(indir):
    """Load a set of storms for processing."""
    
    landfalls = pd.read_csv(f"{indir}/landfalls.csv")
    dfs = []
    for f in sorted(glob.glob(f"{indir}/track*csv")):
        dfs.append(pd.read_csv(f))
    
    maxel = np.load(f"{indir}/outputs/maxele.npz")
    zeta = maxel["zeta"][:]
    zeta_time = maxel["zeta_time"][:]
    zeta[zeta<0] = 0
    storms = []
    for i in range(len(dfs)):
        row = landfalls.iloc[i]
        storms.append(StormData(dfs[i], row, zeta, zeta_time))
    return storms

def latlon_tree(lats, lons):
    return BallTree(latlon_tree_points(lats, lons), metric='haversine')

def latlon_tree_points(lats, lons):
    return np.deg2rad(np.column_stack([lats, lons]))

class VisionFeatures:
    """Class to create gridded feature maps suitable for use in vision models."""

    
    def __init__(
        self,
        bathy,
        lats,
        lons,
        harmonics=None,
        landfall_window = 1.25, # window in degrees about landfall
        temporal_window = [24, 12], # hours before and after
        spatial_points = 128, # resolution in degrees
        temporal_res = 3, # resolution in hours
        segment=True, # whether to segment the data
        segment_zeta_rel_thresh=.8, # threshold as ratio of maximum zeta
        segment_zeta_abs_thresh=2.5, # threshold in absolute terms
        segment_land_dist=10, # max dist in km from land for segment
        treepath = "mesh_tree.pkl" # path to BallTree of mesh coords
    ):
        """Initialize class."""
        self._bathy = bathy
        self._lats = lats
        self._lons = lons
        self._dt = temporal_res
        self._dx = 2*landfall_window / (spatial_points-1)
        self._landfall_window = landfall_window
        self._temporal_window = temporal_window
        self._spatial_points = spatial_points
        self._harmonics = harmonics
        self._segment_zeta_rel_thresh = segment_zeta_rel_thresh
        self._segment = segment
        self._segment_zeta_abs_thresh = segment_zeta_abs_thresh
        self._segment_land_dist = segment_land_dist
        self._init_tree(treepath)
    
    def _init_tree(self, treepath):
        if not os.path.exists(treepath):
            # need the file with mesh coordinates and station coordinates
            tree = latlon_tree(self._lats, self._lons)
            #coords = np.deg2rad(np.column_stack([self._lats, self._lons]))
            #tree = BallTree(coords, metric='haversine')
            with open(treepath, "wb") as fp: pickle.dump(tree, fp)
            self._tree = tree
        else:
            with open(treepath, "rb") as fp:
                self._tree = pickle.load(fp)

    def do_interp(self, lats, lons, arrs):
        """Interpolate a sequence of mesh-based variables to a grid."""
        
        lat_bins = np.linspace(lats[0]-self._dx/2, lats[-1]+self._dx/2, len(lats)+1)
        lon_bins = np.linspace(lons[0]-self._dx/2, lons[-1]+self._dx/2, len(lons)+1)
        # This does entail a full scan of the mesh inds
        # probably a faster way to do this would be to
        # maintain a sorted list of lats/lons
        # and then do an intersection on the matching indices. . . 
        # but that is complicated
        # a ball tree search is another alternative
        inds = np.where(
            (self._lats<lat_bins[-1]) &
            (self._lats>lat_bins[0]) &
            (self._lons<lon_bins[-1]) &
            (self._lons>lon_bins[0]))
        mesh_lats, mesh_lons = self._lats[inds], self._lons[inds]
        # because of the bounding on inds it is guaranteed to
        # fall within the proper window
        lat_inds = np.searchsorted(lat_bins, mesh_lats)-1
        lon_inds = np.searchsorted(lon_bins, mesh_lons)-1

        nlat, nlon = len(lats), len(lons)

        flat_inds = lat_inds * nlon + lon_inds

        counts = np.bincount(
            flat_inds, minlength=nlat * nlon).reshape((nlat, nlon))
        # avoid divide by zero
        empty = counts == 0        
        empty_inds = np.where(empty)
        num_empty = len(empty_inds[0])
        if num_empty > 0:
            query_points = latlon_tree_points(lats[empty_inds[0]], lons[empty_inds[1]])
            query_inds = self._tree.query(query_points, k=1, return_distance=False)
            query_inds = query_inds.flatten()
        # avoid divide by zero
        counts[empty_inds] = -1
        result = {}
        for key, arr in arrs.items():
            means = np.bincount(
                flat_inds, weights=arr[inds], minlength=counts.size
            ).reshape(counts.shape) / counts
            if num_empty > 0:
                means[empty_inds] = arr[query_inds]
            result[key] = means
            result[key+"_mesh"] = arr[inds]
        result["mesh_inds_in_box"] = inds
        result["box_lat_inds"] = lat_inds
        result["box_lon_inds"] = lon_inds
        result["empty_inds"] = empty_inds
        return result
        
        
    def process_storm(self, storm):
        """Given a storm extract features."""

        # determine landfall spatiotemporal window
        landfall_lat = storm.landfall_lat
        landfall_lon = storm.landfall_lon
        if landfall_lon > 180: landfall_lon -= 360
        if landfall_lon < -180: landfall_lon += 360
        window = self._landfall_window
        res = self._spatial_points
        grid_lats = np.linspace(landfall_lat-window, landfall_lat+window, res)
        grid_lons = np.linspace(landfall_lon-window, landfall_lon+window, res)
        # handle wrap-around
        grid_lons[grid_lons >= 180] -= 360
        grid_lons[grid_lons <= -180] += 360
        lons, lats = np.meshgrid(grid_lons, grid_lats)

        before, after = self._temporal_window
        hours = storm.track['tstep'].values*TRACK_DT
        landfall_hour = storm.landfall_time*TRACK_DT
        
        start_time = landfall_hour - before
        stop_time = landfall_hour + after
        hours = hours[(hours>=start_time)&(hours<=stop_time)]
        if not len(hours):
            print("WARNING: no valid times in landfall window!")
            return

        if hours[0] >= hours[-1]:
            print("WARNING: start time in landfall window not less than end time!")
            return
        
        num_times = int((stop_time-start_time)/self._dt) + 1
        times = np.linspace(start_time, stop_time, num_times)
        
        # compute gridded winds and pressure
        
        winds = HollandWinds(storm.track, dt=TRACK_DT)
        windx, windy, pres = winds.evaluate_many(times, lats, lons)        
        
        # bathymetry (and other features need to be interpolated to the grid)
        # we can use binning + averaging (best for high resolution)
        # or nearest neighbor (good for low resolution)
        
        data = {
            "windx": windx,
            "windy": windy,
            "pres": pres,
            "lat": grid_lats,
            "lon": grid_lons,
            "landfall_hour": landfall_hour,
            "landfall_lat": landfall_lat,
            "landfall_lon": landfall_lon,
            "basin": storm.basin_str
        }
        
        arrs_to_interp = {
            "bathy": self._bathy,
            "zeta": storm.zeta,
            "zeta_time": (storm.zeta_time/(24*3600) - landfall_hour/24-7)
        }

        if self._harmonics is not None:
            arrs_to_interp.update(self._harmonics)
        
        interpolated_arrs = self.do_interp(grid_lats, grid_lons, arrs_to_interp)
        # squares that are land and have no mesh points should be
        # assumed to be outside of the ADCIRC mesh, and values should be set to 
        # a default instead of filled in with nearest neighbor interpolation
        # this will fix the error where continents are marked with positive bathymetry
        # and high zeta
        is_land = self._filtered_land_mask(lats, lons)
        # the global land mask is pretty fine-grained
        empty_inds = interpolated_arrs["empty_inds"]
        empty_is_land = is_land[empty_inds]
        off_mesh_inds = np.where(empty_is_land)
        box_off_mesh_inds = (empty_inds[0][off_mesh_inds], empty_inds[1][off_mesh_inds])
        off_mesh_defaults = {
            "bathy": -20,
            "zeta": 0,
            "zeta_time": -12
        }
        for k, v in off_mesh_defaults.items():
            interpolated_arrs[k][box_off_mesh_inds] = v
        data.update(interpolated_arrs)
        data["land_mask"] = is_land
        if self._segment:
            self.add_segmentation(data, is_land)

        return data

    def _filtered_land_mask(self, lats, lons):
        """Filter the raw land mask to cut down on tiny one-pixel islands."""

        land_mask = globe.is_land(lats, lons)

        neighbors = np.zeros(land_mask.shape)
        neighbors[:-1] += land_mask[1:]
        neighbors[1:] += land_mask[:-1]
        neighbors[:, 1:] += land_mask[:, :-1]
        neighbors[:, :-1] += land_mask[:, 1:]        
        return land_mask & (neighbors >= 2)
        
    def add_segmentation(self, data, is_land):
        """Add segmentation masks to the data."""

        # step 1 - determine distance to land for each grid cell
        land_inds = np.where(is_land)
        if not len(land_inds[0]):
            print("Empty land mask!")
            raise ValueError()
        land_lats = data["lat"][land_inds[0]]
        land_lons = data["lon"][land_inds[1]]
        tree = latlon_tree(land_lats, land_lons)
        # include points marked as land by global_land_mask but
        # still inundated - these likely correspond to differences in ADCIRC's mesh
        # and the global land mask
        sea_mask = ~is_land | (data["zeta"] > 0)
        sea_inds = np.where(sea_mask)
        sea_query_points = latlon_tree_points(data["lat"][sea_inds[0]], data["lon"][sea_inds[1]])
        land_dist, _ = tree.query(sea_query_points, k=1, return_distance=True)
        land_dist = earth_radius * land_dist.flatten()
        coastal_inds = np.where(land_dist < self._segment_land_dist)

        coastal_mask = np.zeros_like(sea_mask)
        coastal_mask[sea_inds[0][coastal_inds], sea_inds[1][coastal_inds]] = True

        data["coastal_mask"] = coastal_mask

        # step 2, apply zeta mask on top of coastal mask
        max_zeta = data["zeta"][coastal_mask].max()
        zeta_rel_thresh = max_zeta * self._segment_zeta_rel_thresh
        thresh = min(zeta_rel_thresh, self._segment_zeta_abs_thresh)
        data["zeta_mask"] = (data["zeta"] > thresh) & coastal_mask


def make_vision_features(basedir, **kwargs):    
    df = pd.read_csv(basedir+"/global_mesh_coords.csv", index_col=0)
    with h5py.File(basedir+"/global_bathy.hdf5") as ds:
        bathy = ds["depth"][:]

    harmonics_arrs = None
    #harmonics_arrs = {}
    #with h5py.File(basedir+"/global_tidal_amplitudes.hdf5", "r") as harmonics:
    #    for k in sorted(list(harmonics.keys())):
    #        harmonics_arrs[k] = harmonics[k][:]

    return VisionFeatures(bathy, df["lat"].values, df["lon"].values, harmonics=harmonics_arrs, **kwargs)

def create_dataset(runsdir, outputdir, basedir="/work2/08009/bpachev/ls6/simulations/global-ml", **kwargs):
    """Process a set of ADCIRC runs and create a gridded dataset."""

    rundirs = sorted(list(glob.glob(runsdir+"/*/")))
    os.makedirs(outputdir, exist_ok=True)
    vf = make_vision_features(basedir, **kwargs)

    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    if rank == 0: print(f"Processing {len(rundirs)} ADCIRC runs")
    for rundir in rundirs[rank::size]:
        #print(rundir)
        storms = load_storms(rundir)
        for i, storm in enumerate(storms):
            try:
                data = vf.process_storm(storm)
            except ValueError:
                continue
            outfname = outputdir + "/" + rundir.strip("/").split("/")[-1] + f"_{i}.hdf5"
            with h5py.File(outfname, "w") as ds:
                for k, arr in data.items():
                    ds[k] = arr


if __name__ == "__main__":
    """
    basedir = "/work2/08009/bpachev/ls6/simulations/global-ml"
    vf = make_vision_features(basedir)
    #storms = load_storms(basedir+"/new_packed_inputs_normal/run00")
    storms = load_storms("/scratch1/08009/bpachev/global_tcs_v2/runs0_99/run0000/")
    for storm in storms:
        vf.process_storm(storm)
    """

    create_dataset(
            "/scratch/08009/bpachev/global_tcs_v2/",
            "/scratch/08009/bpachev/global_tcs_datasets/full_segmented",
            segment_land_dist=5
    )
