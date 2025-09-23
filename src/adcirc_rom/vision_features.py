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

# TODO - directly process a packed input with multiple storms
# TODO - add functions for visualization of gridded features
# TODO - get gridded winds/bathymetry/pressure time series as direct inputs
# TODO - analyze time of zeta max (relative to time of landfall)
# TODO - get bathymetry gradients
TRACK_DT = 3

class StormData:
    
    def __init__(self, track, landfall, zeta, zeta_time):
        self.track = track
        self.landfall_row = landfall
        self.zeta = zeta
        self.zeta_time = zeta_time
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
        print("no landfall according to global_land_mask")
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
        
class VisionFeatures:
    """Class to create gridded feature maps suitable for use in vision models."""

    
    def __init__(
        self,
        bathy,
        lats,
        lons,
        landfall_window = 2.5, # window in degrees about landfall
        temporal_window = [24, 12], # hours before and after
        spatial_res = .02, # resolution in degrees
        temporal_res = 3, # resolution in hours
        treepath = "mesh_tree.pkl" # path to BallTree of mesh coords
    ):
        """Initialize class."""
        self._bathy = bathy
        self._lats = lats
        self._lons = lons
        self._dt = temporal_res
        self._dx = spatial_res
        self._landfall_window = landfall_window
        self._temporal_window = temporal_window
        self._spatial_points = int(2*landfall_window/spatial_res) + 1

        self._init_tree(treepath)
    
    def _init_tree(self, treepath):
        if not os.path.exists(treepath):
            # need the file with mesh coordinates and station coordinates
            coords = np.deg2rad(np.column_stack([self._lats, self._lons]))
            tree = BallTree(coords, metric='haversine')
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
            query_points = np.deg2rad(
                np.column_stack(
                    [lats[empty_inds[0]], lons[empty_inds[1]]]
                )
            )
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
        window = self._landfall_window
        res = self._spatial_points
        grid_lats = np.linspace(landfall_lat-window, landfall_lat+window, res)
        grid_lons = np.linspace(landfall_lon-window, landfall_lon+window, res)
        lons, lats = np.meshgrid(grid_lons, grid_lats)

        before, after = self._temporal_window
        hours = storm.track['tstep'].values*TRACK_DT
        landfall_hour = storm.landfall_time*TRACK_DT
        
        hours = hours[(hours>=landfall_hour-before)&(hours<=landfall_hour+after)]
        if not len(hours):
            print("WARNING: no valid times in landfall window!")
            return

        start_time, stop_time = hours[0], hours[-1]
        if start_time >= stop_time:
            print("WARNING: start time in landfall window not less than end time!")
            return
        
        num_times = int((stop_time-start_time)/self._dt) + 1
        times = np.linspace(start_time, start_time, num_times)
        
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
            "landfall_lon": landfall_lon
        }
        
        arrs_to_interp = {
            "bathy": self._bathy,
            "zeta": storm.zeta,
            "zeta_time": (storm.zeta_time/(24*3600) - landfall_hour/24-7)
        }
        
        interpolated_arrs = self.do_interp(grid_lats, grid_lons, arrs_to_interp)
        # squares that are land and have no mesh points should be
        # assumed to be outside of the ADCIRC mesh, and values should be set to 
        # a default instead of filled in with nearest neighbor interpolation
        # this will fix the error where continents are marked with positive bathymetry
        # and high zeta
        is_land = globe.is_land(lats, lons)
        empty_inds = interpolated_arrs["empty_inds"]
        empty_is_land = is_land[empty_inds]
        off_mesh_inds = np.where(empty_is_land)
        box_off_mesh_inds = (empty_inds[0][off_mesh_inds], empty_inds[1][off_mesh_inds])
        off_mesh_defaults = {
            "bathy": -20,
            "zeta": 0,
            "zeta_time": -12
        }
        print(is_land, empty_is_land, off_mesh_inds)
        for k, v in off_mesh_defaults.items():
            interpolated_arrs[k][box_off_mesh_inds] = v
        data.update(interpolated_arrs)
        return data

def make_vision_features(basedir, **kwargs):    
    df = pd.read_csv(basedir+"/global_mesh_coords.csv", index_col=0)
    with h5py.File(basedir+"/global_bathy.hdf5") as ds:
        bathy = ds["depth"][:]
    
    return VisionFeatures(bathy, df["lat"].values, df["lon"].values, **kwargs)
    
if __name__ == "__main__":
    basedir = "/work2/08009/bpachev/ls6/simulations/global-ml"
    vf = make_vision_features(basedir)
    #storms = load_storms(basedir+"/new_packed_inputs_normal/run00")
    storms = load_storms("/scratch1/08009/bpachev/global_tcs_v2/runs0_99/run0000/")
    for storm in storms:
        vf.process_storm(storm)