from mpi4py import MPI
import glob
import h5py
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
import fire
import gc

filters = {"max": maximum_filter, "mean": uniform_filter, "min": minimum_filter}

def compute_spatial_stats(data, windows=[2,4,8,16]):
    """Compute spatial kernels on given data."""

    out = {}
    for window in windows:
        for filter_name, filter_func in filters.items():
            arr = np.empty_like(data)
            if len(data.shape) > 2:
                for i in range(len(data)):
                    arr[i] = filter_func(data[i], window)
            else:
                arr[:] = filter_func(data, window)                
            out[f"{filter_name}_{window}"] = arr
    return out

def process_file(fname):
    features = {}
    gridded_features = {}
    forcing_arrs = ["pres", "windx", "windy"]
    static_arrs = ["land_mask", "bathy"]
    stat_funcs = {"min": np.min, "max": np.max, "mean": np.mean}
    with h5py.File(fname) as ds:
        mask = ds["coastal_mask"][:]
        for arr in forcing_arrs:
            temporal_data = ds[arr][:]
            spatial_stats = compute_spatial_stats(temporal_data)
            spatial_stats["base"] = temporal_data
            for k, v in spatial_stats.items():
                for name, func in stat_funcs.items():
                    feat_name = f"{arr}_{name}_{k}"
                    gridded_features[feat_name] = func(v, axis=0)

        for arr in static_arrs:
            static_data = ds[arr][:]
            spatial_stats = compute_spatial_stats(static_data)
            gridded_features[arr] = static_data
            for k, v in spatial_stats.items():
                gridded_features[f"{arr}_{k}"] = v

        # metadata
        x, y =  np.meshgrid(ds["lon"][:], ds["lat"][:])
        gridded_features["x"] = x
        gridded_features["y"] = y
        gridded_features["maxele"] = ds["zeta"][:]
        
        return {name: gridded_arr[mask] for name, gridded_arr in gridded_features.items()}
    
def vision_dataset_to_xgb(indir, outfile):
    """Given a masked vision dataset, convert it to a traditional grid-encoded XGB dataset."""

    datafiles = sorted(list(glob.glob(indir+"/*hdf5")))

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    arrs = {}
    
    for i in range(rank, len(datafiles), size):
        fname = datafiles[i]
        features = process_file(fname)
        npoints = len(features["maxele"])
        if npoints == 0:
            print(f"Warning: no points for {fname}")
            continue
        if rank == 0:
            print("Points", npoints)

        features["storm"] = np.ones(npoints, dtype=np.int32) * i
        
        if not len(arrs):
            for k in features:
                arrs[k] = [features[k]]
        else:
            for k in features:
                arrs[k].append(features[k])

    local_data = {k: np.concatenate(arrs[k]) for k in arrs}
    data = {}
    keys = sorted(list(local_data.keys()))
    root = 0
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

    if rank != root: return
        
    with h5py.File(outfile, "w") as ds:
        for k in data:
            ds[k] = data[k]
        ds["storm_names"] = datafiles

if __name__ == "__main__":
    fire.Fire(vision_dataset_to_xgb)
