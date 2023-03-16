import os
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import h5py
import netCDF4 as nc
import numpy as np
import xgboost as xgb
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

from adcirc_rom.constants import earth_radius


class FeatureImportance(xgb.callback.TrainingCallback):
    """Monitors the feature importances during cross-validation."""

    def __init__(self, sample_rounds=50):
        """sample_rouds defines how often we sample the feature importances"""

        self.feature_importances = defaultdict(list)
        self.sample_rounds = sample_rounds

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.sample_rounds:
            return
        bst = model.cvfolds[0].bst
        self.add_importances(bst)

    def add_importances(self, bst):
        importances = bst.get_score(importance_type="gain")
        for k, v in importances.items():
            self.feature_importances[k].append(v)

    def print_summary(self, top=10):
        feat_names = []
        avg_importances = []
        for k, v in self.feature_importances.items():
            feat_names.append(k)
            avg_importances.append(np.mean(np.array(v)))
        avg_importances = np.array(avg_importances)
        order = np.argsort(avg_importances)
        print(f"Top {top} Features")
        for i, ind in enumerate(order[-1 : -top - 1 : -1]):
            print(f"#{i+1}: {feat_names[ind]} with gain {avg_importances[ind]:.2f}")


def extract_features(ds, include_latlon=False, exclude_bathy=False):
    """
    Extracts features from an xarray dataset based on certain criteria.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset.
    include_latlon : bool, optional
        Whether to include variables with names "x" and "y", by default False.
    exclude_bathy : bool, optional
        Whether to exclude variables that start with "bathy", by default False.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple containing a NumPy array of the extracted features (with dimensions
        `(number of samples, number of features)`) and a list of their names.
    """
    arrs = []
    names = []

    # add scalar properties
    for var in sorted(list(ds.keys())):
        # if var.count("_") >= 3: continue
        if (
            (not exclude_bathy and var.startswith("bathy"))
            or var in ["coastal_dist", "landfall_dist", "depth"]
            or (include_latlon and var in ["x", "y"])
            or var.startswith("amplitude")
            or any([k in var for k in ["wind", "pressure", "magnitude", "iceaf"]])
        ):
            names.append(var)

    n = ds[names[0]].shape[0]
    mat = np.zeros((len(names), n))
    for i, name in enumerate(names):
        mat[i] = ds[name][:]

    return mat.T, names


class CorrelationFilter:
    """A class to eliminate correlated features"""

    def __init__(self, threshold):
        """Initialize the filter"""

        self.threshold = threshold

    def fit(self, X, y=None):
        """Determine the features to remove"""

        corrs = np.corrcoef(X, rowvar=False)
        n = corrs.shape[0]
        keep = np.ones(n, dtype=bool)

        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                if abs(corrs[i, j]) > self.threshold:
                    keep[j] = 0

        self.cols_to_keep = np.where(keep)[0]

    def transform(self, X):
        return X[:, self.cols_to_keep]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class FeatureImportanceFilter:
    """Determine feature importances by training a regressor on a subsample of the data"""

    def __init__(self, max_features=50):
        """Initialize the transform and set the max number of features to keep"""

        self.max_features = max_features

    def fit(self, X, y):
        model = xgb.XGBRegressor(
            n_estimators=10, importance_type="gain", n_jobs=32, subsample=0.5
        )
        model.fit(X, y)
        importances = model.feature_importances_
        # select the top features based on importance
        inds = np.argsort(importances)
        self.cols_to_keep = inds[-self.max_features :]

    def transform(self, X):
        return X[:, self.cols_to_keep]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def save_stats(stats, outfile):
    with h5py.File(outfile, "w") as outf:
        for k, v in stats.items():
            outf[k] = v


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

        # self.counts = np.histogram2d(self.x, self.y, bins=[self.x_bins, self.y_bins])[0]
        # avoid divide by zero
        self.counts[self.counts == 0] = 1

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

        if name is not None:
            pref = name + "_"
        else:
            pref = ""

        for s in scales:
            suf = self.resolution * s
            res[f"{pref}mean_{suf}"] = uniform_filter(means, s)[(outx_inds, outy_inds)]
            if s == 1:
                continue
            res[f"{pref}max_{suf}"] = maximum_filter(means, s)[(outx_inds, outy_inds)]
            res[f"{pref}min_{suf}"] = minimum_filter(means, s)[(outx_inds, outy_inds)]

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
