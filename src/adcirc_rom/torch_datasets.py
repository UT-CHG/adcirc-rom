import torch
from torch.utils.data import Dataset
import os
import glob
import h5py
import numpy as np

class SyntheticTCDataset(Dataset):
    """A dataset of synthetic tc simulations
    Currently meant for a single basin
    """

    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1

    def __init__(self, folder, val=False, test=False):
        self.folder = folder
        self.val = val
        self.test = test

        files = sorted(list(glob.glob(folder+"/*/*.hdf5")), key=lambda fname: fname.split("/")[-1])
        split_ind_val = int((1 - self.VAL_SPLIT - self.TEST_SPLIT) * len(files))
        split_ind_test = int((1 - self.TEST_SPLIT) * len(files))
        if test:
            self.files = files[split_ind_test:]
        elif val:
            self.files = files[split_ind_val:split_ind_test]
        else:
            self.files = files[:split_ind_val]

        print(f"Initializing SyntheticTCDataset with val={val}, test={test}, nfiles={len(self.files)}.")
    

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

        with h5py.File(fname, 'r') as ds:
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
