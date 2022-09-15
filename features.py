import netCDF4 as nc
import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
import h5py
import os
import geopandas as gpd

earth_radius = 6731

def save_stats(stats, outfile):
    with h5py.File(outfile, "w") as outf:
        for k,v in stats.items():
            outf[k] = v
        

class GridEncoder:
    """A class to represent spatial aggregations of mesh variables like bathymetry, wind, pressure, etc.
    """

    def __init__(self, x, y, resolution=0.01, bounds=None):
        """Initialize the encoder
 
        Parameters
        ----------
        resolution (float) - the cell resolution in degrees
        bounds (tuple) (min_lat, max_lat, min_lon, max_lon) defining the region of interest (for efficiency)
        """

        self.x, self.y = x, y
        
        if bounds is None:
            min_lat, max_lat, min_lon, max_lon = self.y.min(), self.y.max(), self.x.min(), self.x.max()
        else:
            min_lat, max_lat, min_lon, max_lon = bounds

        # Make bins
        self.x_bins = np.arange(min_lon, max_lon+resolution, resolution)
        self.y_bins = np.arange(min_lat, max_lat+resolution, resolution)
        self.resolution = resolution

        self.x_inds = np.searchsorted(self.x_bins, self.x)
        self.y_inds = np.searchsorted(self.y_bins, self.y)
        
        nx, ny = len(self.x_bins)+1, len(self.y_bins)+1
        
        self.flat_inds = self.x_inds * ny + self.y_inds
        
        self.counts = np.bincount(self.flat_inds, minlength=nx*ny).reshape((nx,ny))
        
        #self.counts = np.histogram2d(self.x, self.y, bins=[self.x_bins, self.y_bins])[0]
        #avoid divide by zero
        self.counts[self.counts == 0] = 1
        

    def encode(self, var, scales=[1], outx=None, outy=None, name=None):
        """Divide the mesh into cells and determine the min, max, and mean bathymetry within each cell
        
        Parameters
        ----------
        scale - over how many cells to take the aggregates
        """


        means = np.bincount(self.flat_inds,
                            weights=var,
                            minlength=self.counts.size).reshape(self.counts.shape)
        
        if outx is not None and outy is not None:
            outx_inds = np.searchsorted(self.x_bins, outx)
            outy_inds = np.searchsorted(self.y_bins, outy)
        else:
            outx_inds, outy_inds = self.x_inds, self.y_inds
        
        means /= self.counts
        nx, ny = means.shape
        outx_inds[outx_inds >= nx] = nx-1
        outy_inds[outy_inds >= ny] = ny-1
        res = {}
        
        if name is not None:
            pref = name+"_"
        else:
            pref = ""
        
        for s in scales:
            suf = self.resolution * s
            res[f"{pref}mean_{suf}"] = uniform_filter(means, s)[(outx_inds, outy_inds)]
            if s == 1: continue
            res[f"{pref}max_{suf}"] = maximum_filter(means, s)[(outx_inds, outy_inds)]
            res[f"{pref}min_{suf}"] = minimum_filter(means, s)[(outx_inds, outy_inds)]
            
        return res        

def init_shared_features(input_dir, output_dir, scales=[5, 10, 40, 100]):
    """Initialize shared features such as bathy_stats and coastal distances

    input_dir must contain a maxele.63.nc file with grid information
    the files bathy_stats.hdf5 and coastal_dists.hdf5 will be output to output_dir
    """
    
    with nc.Dataset(datadir+"/maxele.63.nc") as ds:

        enc = GridEncoder(ds["x"][:], ds["y"][:],
                        resolution=.01,     
                        bounds= (24,32,-98, -88))

        depth = ds["depth"][:]
        stats = enc.encode(depth, scales=scales, name="bathy")
        save_stats(stats, "data/bathy_stats.hdf5")
        x, y = ds["x"][:], ds["y"][:]

        shoreline = gpd.read_file("Gulf_of_Mexico_GCOOS_Region_with_GSHHS_shorelines__GCOOS_.geojson")

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
        R = 6731 # radius of earth in km
        dist, ind = tree.query(np.column_stack([np.deg2rad(y), np.deg2rad(x)]))
        dist, ind = dist.flatten(), ind.flatten()
        dist *= R

    with h5py.File(output_dir"/coastal_dist.hdf5", "w") as outds:
        outds["dist"] = dist
