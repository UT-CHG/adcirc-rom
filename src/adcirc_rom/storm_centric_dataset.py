from fire import Fire
import glob
import h5py
import pandas as pd
import multiprocessing

def extract_rec(fname):
    """Convert a storm to a single row of features."""
    print("Processing ", fname)
    ds = h5py.File(fname)
    keys = ds.keys()
    basin = fname.split("/")[-3]
    category = int(fname.split("/")[-2][-1])
    rec = {'max_zeta_diff' : ds['zeta_diff'][:].max(), 'category': category, 'basin': basin}
    for k in ds.keys():
        if 'zeta' not in k:
            rec["min_"+k] = ds[k][:].min()
            rec["max_"+k] = ds[k][:].max()
            rec["mean_"+k] = ds[k][:].mean()
    return rec

def main(dirname="/scratch/06307/clos21/public/prateek-updated/single_storm_dataset_fixed"):
    """Make a dataset centered on storms."""

    recs = []
    fnames = sorted(list(glob.glob(dirname+"/*/cat*/*hdf5")))
    with multiprocessing.Pool(32) as pool:
        recs = pool.map(extract_rec, fnames)
    
    df = pd.DataFrame(recs)
    df.to_csv("storm_centric_dataset.csv", index=False)

if __name__ == "__main__":
  Fire(main)
