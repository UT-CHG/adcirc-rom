import numpy as np 
import glob
import h5py
import pandas as pd
import math
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine_vector

from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

import matplotlib.pyplot as plt

import os


# -------------------------------------------------------------------------
# Const
# -------------------------------------------------------------------------
#nm2m=1852. # 1 nautical mile to meters
#kt2ms=nm2m/3600.  # knots to m/s
omega=2*np.pi/(3600.*24. + .2) # angular speed omega=2pi*f(=frequency of earth : 1 cycle per day) 2pi* 1 / day in seconds
#rhoa=1.15 #air density  Kg/m^3
rhoa = 1.293
rhowat0 = 1e3
G = 9.80665
radius=6378388 #137. # earth's radius according to WGS 84
deg2m=np.pi*radius/180.  # ds on cicle equals ds=r*dth - dth=pi/180
one2ten=0.8928  # coefficient for going from 1m to 10m in velocities
BLAdj=0.9
pback = 1013


class HollandWinds:
    """A Python class for generating winds with the symmetric Holland model
    """

    def __init__(self, track, dt=1):
        """Initialize the wind model

        Parameters
        ----------
        track - pd.DataFrame
            A Pandas DataFrame with best track information.
            Must contain the columns: lat, lon, vmax, rmax, minpres
        dt - timedelta in hours
        """

        self.lat = track["lat"].values
        self.lon = track["lon"].values
        self.lon[self.lon > 180] -= 360
        # convert vmax from 10-minute to 1-minute
        self.vmax = track["vmax"].values / one2ten
        self.rmax = track["rmax"].values
        self.minpres = track["minpres"].values
        self.dt = dt
        # compute translational velocities
        self._compute_translational_vel()

    def _compute_translational_vel(self):
        x = self.lon
        y= self.lat
        # convert timestep to seconds
        dt = 3600*self.dt
    
        velocity = np.zeros((len(x), 2))
        velocity[1:, 0] = np.cos(np.deg2rad((y[1:]-y[:-1])/2)) * (x[1:]-x[:-1]) / dt
        velocity[1:, 1] = (y[1:]-y[:-1]) / dt
        velocity[0] = velocity[1]
        self.vtrx = velocity[:,0] * deg2m  #adjust for latitude
        self.vtry = velocity[:,1] * deg2m    
        self.vtr = np.sqrt(self.vtrx**2+self.vtry**2)

    def evaluate(self, t, lats, lons):
        """Evaluate the model at a given time and set of coordinates

        Parameters
        ----------
        t - time in hours
        lats - latitudes to evaluate at
        lons - longitudes to evaluate at
        
        Returns
        ----------
        windx, windy, pres
        """
        # do interpolation
        ind = t/self.dt
        if ind < 0 or ind > len(self.lat):
            # out of bounds - zero winds and background pressure
            return np.zeros_like(lats), np.zeros_like(lats), np.full_like(lats, pback)

        last = int(math.floor(ind))
        curr = int(math.ceil(ind))
        lam = ind-last
        vmax = (1-lam) * self.vmax[last] + lam * self.vmax[curr]
        rmaxh = (1-lam) * self.rmax[last] + lam * self.rmax[curr]
        lat0 = (1-lam) * self.lat[last] + lam * self.lat[curr]
        lon0 = (1-lam) * self.lon[last] + lam * self.lon[curr]
        pc = (1-lam) * self.minpres[last] + lam * self.minpres[curr]
        vtx = (1-lam) * self.vtrx[last] + lam * self.vtrx[curr]
        vty = (1-lam) * self.vtry[last] + lam * self.vtry[curr]
        
        coors = [x for x in zip(lats,lons)]
        r = haversine_vector(coors, len(coors)*[(lat0, lon0)]) * 1000
        # pressure deficit
        DP = (pback-pc)*100
        DP = max(DP, 100)
        vtr = (vtx**2 + vty**2) ** .5
        # need to subtract the translational speed from the vmax
        vmax -= vtr
        vmax /= BLAdj
        bh = (rhoa*np.exp(1)/DP)*(vmax)**2
        # cap bh to be within proper ranges
        bh = min(2.5, max(bh, 1))
        theta=np.arctan2(np.deg2rad(lats-lat0),np.deg2rad(lons-lon0))
        fcor = 2*omega*np.sin(np.deg2rad(lats)) #coriolis force
        r_nd = (rmaxh*1e3/r)**bh
        ur = (
            r_nd * np.exp(1-r_nd) * vmax**2
            + (r*fcor/2)**2
        )**0.5 - r*abs(fcor)/2
        pres_prof = pc+DP*np.exp(-r_nd**bh)/100
        ux = -ur*np.sin(theta) * BLAdj * one2ten
        uy = ur*np.cos(theta) * BLAdj * one2ten
        mul = np.abs(ur)/vmax
        return ux + mul * vtx, uy + mul * vty, pres_prof

    
class GridEncoder:
    """A class to represent spatial aggregations of mesh variables like bathymetry, wind, pressure, etc."""

    def __init__(self, x, y, resolution=0.01, bounds=None):
        """Initialize the encoder

        Parameters
        ----------
        resolution (float) - the cell resolution in degrees
        bounds (tuple) (min_lat, max_lat, min_lon, max_lon) defining the region of interest (for efficiency)
        """

        self.x, self.y = x, y

        if bounds is None:
            min_lat, max_lat, min_lon, max_lon = (
                self.y.min(),
                self.y.max(),
                self.x.min(),
                self.x.max(),
            )
            print("Encoder Bounds", min_lat, max_lat, min_lon, max_lon)
        else:
            min_lat, max_lat, min_lon, max_lon = bounds

        # Make bins
        self.x_bins = np.arange(min_lon, max_lon + resolution, resolution)
        self.y_bins = np.arange(min_lat, max_lat + resolution, resolution)
        self.resolution = resolution

        self.x_inds = np.searchsorted(self.x_bins, self.x)
        self.y_inds = np.searchsorted(self.y_bins, self.y)

        nx, ny = len(self.x_bins) + 1, len(self.y_bins) + 1

        self.flat_inds = self.x_inds * ny + self.y_inds

        self.counts = np.bincount(self.flat_inds, minlength=nx * ny).reshape((nx, ny))
        # avoid divide by zero
        self.empty = self.counts == 0
        self.counts[self.empty] = 1

    def encode(self, var, scales=[1], outx=None, outy=None, name=None):
        """Divide the mesh into cells and determine the min, max, and mean bathymetry within each cell

        Parameters
        ----------
        scale - over how many cells to take the aggregates
        """

        means = np.bincount(
            self.flat_inds, weights=var, minlength=self.counts.size
        ).reshape(self.counts.shape)

        if outx is not None and outy is not None:
            outx_inds = np.searchsorted(self.x_bins, outx)
            outy_inds = np.searchsorted(self.y_bins, outy)
        else:
            outx_inds, outy_inds = self.x_inds, self.y_inds

        means /= self.counts
        nx, ny = means.shape
        outx_inds[outx_inds >= nx] = nx - 1
        outy_inds[outy_inds >= ny] = ny - 1
        res = {}
        # properly handle empty grid cells for min/max filters
        max_input = means.copy()
        max_input[self.empty] = -np.inf
        min_input = means.copy()
        min_input[self.empty] = np.inf
        if name is not None:
            pref = name + "_"
        else:
            pref = ""

        for s in scales:
            suf = self.resolution * s
            # Adjust for 
            empty_fraction = uniform_filter(self.empty.astype(float), s)[(outx_inds, outy_inds)]
            empty_fraction[empty_fraction == 1] = 0
            res[f"{pref}mean_{suf}"] = uniform_filter(means, s)[(outx_inds, outy_inds)] / (1-empty_fraction)
            if s == 1:
                continue
            res[f"{pref}max_{suf}"] = maximum_filter(max_input, s)[(outx_inds, outy_inds)]
            res[f"{pref}min_{suf}"] = minimum_filter(min_input, s)[(outx_inds, outy_inds)]

        return res


def init_shared_features(
    input_dir,
    output_dir="data",
    scales=[5, 10, 40, 100],
    bounds=(24, 32, -98, -88),
    coastfile="gulf_coast.geojson",
):
    """Initialize shared features such as bathy_stats and coastal distances

    input_dir must contain a maxele.63.nc file with grid information
    the files bathy_stats.hdf5 and coastal_dists.hdf5 will be output to output_dir
    """

    with nc.Dataset(input_dir + "/maxele.63.nc") as ds:
        enc = GridEncoder(ds["x"][:], ds["y"][:], resolution=0.01, bounds=bounds)

        depth = ds["depth"][:]
        stats = enc.encode(depth, scales=scales, name="bathy")
        save_stats(stats, output_dir + "/bathy_stats.hdf5")
        x, y = ds["x"][:], ds["y"][:]

        shoreline = gpd.read_file(str(Path(__file__).parent / coastfile))

        lats = []
        lons = []

        for geom in shoreline.iloc[0]["geometry"].geoms:
            for coord in geom.exterior.coords:
                lons.append(coord[0])
                lats.append(coord[1])
            for interior in geom.interiors:
                for coord in interior.coords:
                    lons.append(coord[0])
                    lats.append(coord[1])

        lons, lats = np.deg2rad(lons), np.deg2rad(lats)
        from sklearn.neighbors import BallTree

        tree = BallTree(np.column_stack([lats, lons]), metric="haversine")
        dist, ind = tree.query(np.column_stack([np.deg2rad(y), np.deg2rad(x)]))
        dist, ind = dist.flatten(), ind.flatten()
        dist *= earth_radius

    with h5py.File(output_dir + "/coastal_dist.hdf5", "w") as outds:
        outds["dist"] = dist
        
        
# def save_stats(stats, outfile):
#     with h5py.File(outfile, "a") as outf:
#         for k, v in stats.items():
#             try:
#                 outf[k] = v

def save_stats(stats, outfile):
    with h5py.File(outfile, "a") as outf:
        for k, v in stats.items():
            # Check if the dataset already exists
            if k in outf:
                # If it exists, delete the existing dataset
                del outf[k]
            # Create a new dataset
            outf.create_dataset(k, data=v, compression ='gzip')
