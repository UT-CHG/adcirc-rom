import netCDF4 as nc
import numpy as np
from sklearn.neighbors import BallTree
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter, median_filter
import h5py

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

        if bounds is None:
            min_lat, max_lat, min_lon, max_lon = self.y.min(), self.y.max(), self.x.min(), self.x.max()
        else:
            min_lat, max_lat, min_lon, max_lon = bounds

        # Make bins
        self.x_bins = np.arange(min_lon, max_lon+resolution, resolution)
        self.y_bins = np.arange(min_lat, max_lat+resolution, resolution)

        self.x_inds = np.searchsorted(self.x_bins, self.x)
        self.y_inds = np.searchsorted(self.y_bins, self.y)
        
        nx, ny = len(self.x_bins) + 1, len(self.y_bins) + 1
        

    def encode(self, var, scales=[1], outx=None, outy=None):
        pass
        
class BathyEncoder:
    """A class to encode bathymetry data in a non-local fashion
    """

    def __init__(self, fname):
        """Initialize the encoder
        """
        
        with nc.Dataset(fname) as ds:
            self.x = ds["x"][:]
            self.y = ds["y"][:]
            self.depth = ds["depth"][:]


    def resolution(self):
        """Determine distance to nearest neighbor
        """

        self.coords = np.deg2rad(np.column_stack([self.y, self.x]))
        self.tree = BallTree(self.coords, metric="haversine")
        
        dist, ind = self.tree.query(self.coords[::100], k=2)
        print(dist[:,1]*earth_radius, ind[:,1])


    # How to differentiate islands from regular land?
    # Will 
        
    def bathy_stats(self, resolution=.05, bounds=None, scales=[1], otherx=None, othery=None):
        """Divide the mesh into cells and determine the min, max, and mean bathymetry within each cell
        
        Parameters
        ----------
        scale - over how many cells to take the aggregates
        """

        
        counts = np.zeros((nx, ny))
        means = np.zeros((nx, ny))
        mins = np.full((nx, ny), np.inf)
        maxes = np.full((nx, ny), -np.inf)
        
        for i, j in zip(x_inds, y_inds):
            d = self.depth[j]
            counts[i,j] += 1
            means[i,j] += d
            maxes[i,j] = max(maxes[i,j], d)
            mins[i,j] = min(mins[i,j], d)
        
        counts[counts==0] = 1
        means /= counts
        
        if otherx is not None and othery is not None:
            outx_inds = np.searchsorted(x_bins, otherx)
            outy_inds = np.searchsorted(y_bins, othery)
        else:
            outx_inds, outy_inds = x_inds, y_inds
        
        res = {}
        
        for s in scales:
            suf = resolution * s
            res[f"bathy_mean_{suf}"] = uniform_filter(means, s)[(outx_inds, outy_inds)]
            res[f"bathy_median_{suf}"] = median_filter(means, s)[(outx_inds, outy_inds)]
            res[f"bathy_max_{suf}"] = maximum_filter(maxes, s)[(outx_inds, outy_inds)]
            res[f"bathy_min_{suf}"] = minimum_filter(mins, s)[(outx_inds, outy_inds)]
            
        return res

if __name__ == "__main__":

    enc = BathyEncoder("data/storms/s001/maxele.63.nc")
    params = {
        "resolution": 0.01,
        "bounds": (24,32,-98, -88),
        "scales": [10, 40, 100]
    }
    
    stats = enc.bathy_stats(**params)
    save_stats(stats, "data/bathy_stats.hdf5")

    with nc.Dataset("data/historical_storms/ike/maxele.63.nc") as ike_ds:
        ike_x = ike_ds["x"][:]
        ike_y = ike_ds["y"][:]
        
    ike_stats = enc.bathy_stats(otherx = ike_x, othery = ike_y, **params)
    save_stats(ike_stats, "data/historical_storms/ike/bathy_stats.nc")