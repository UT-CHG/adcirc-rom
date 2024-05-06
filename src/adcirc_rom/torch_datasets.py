import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import h5py
import numpy as np

class SyntheticTCDataset(Dataset):
    """A dataset of synthetic tc simulations

    Currently meant for a single basin
    """

    VAL_SPLIT=.2


    def __init__(self, folder, val=False):
        """Initialize the dataset from the source folder, given the rank and world size
        """

        self.folder = folder
        self.val = val

        files = sorted(list(glob.glob(folder+"/*/*.hdf5")), key=lambda fname: fname.split("/")[-1])
        split_ind = int((1-self.VAL_SPLIT) * len(files))
        if val:
            self.files = files[split_ind:]
        else:
            self.files = files[:split_ind]

        print(f"Initialzing SyntheticTCDataset with val={val}, nfiles = {len(self.files)}.")

    def __len__(self):
        """Return length of dataset
        """

        return len(self.files)

    def __getitem__(self, idx):
        """Access an item at a given index
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) > 1: raise RuntimeWarning("Unable to handle idx of length > 1")
            idx = idx[0]

        fname = self.files[idx]

        with h5py.File(fname) as ds:
            keys = sorted(list(k for k in ds.keys() if k != 'zeta_max'))
            zeta = ds['zeta_max'][:]
            mat = np.empty((len(zeta), len(keys)))
            for i, k in enumerate(keys):
                mat[:, i] = ds[k][:]

            return {
                    'zeta_max': torch.Tensor(ds['zeta_max'][:]),
                    'features': torch.Tensor(mat)
            }

def tc_collate_fn(samples):
    """Collate a list of samples
    """

    zetas = []
    features = []
    for s in samples:
        zetas.append(s['zeta_max'])
        features.append(s['features'])

    return torch.cat(zetas), torch.cat(features)


if __name__ == "__main__":

    ds = SyntheticTCDataset("/scratch/06307/clos21/shared/prateek/NA/")
    storms = []
    for i in range(1, 100, 10):
        storms.append(ds[i])
    zeta, feats = tc_collate_fn(storms)
    print(zeta.shape, feats.shape)
