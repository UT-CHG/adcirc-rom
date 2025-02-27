import torch
from torch.utils.data import Dataset
import os
import glob
import h5py
import numpy as np
import random

class SyntheticTCDataset(Dataset):
    """A dataset of synthetic tc simulations
    Currently meant for a single basin
    """

    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    UNREADABLE_FILES = ['107.hdf5', '160.hdf5', '304.hdf5', '354.hdf5', '543.hdf5']

    def __init__(self, folder, val=False, test=False, seed=36):
        self.folder = folder
        self.val = val
        self.test = test

        random.seed(seed)
        np.random.seed(seed)
        
        #files = sorted(list(glob.glob(folder+"/*/*.hdf5")), key=lambda fname: fname.split("/")[-1])
        files = sorted(list(glob.glob(folder+"/*.hdf5")), key=lambda fname: fname.split("/")[-1])
        files = [f for f in files if not any(unreadable in f for unreadable in self.UNREADABLE_FILES)]

        random.shuffle(files)

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
        
        top_features = ['bathy_mean_0.05', 'bathy', 'max_pres_min_0.4', 'min_windy_min_0.05', 
            'mean_winds_mean_0.1', 'min_winds_mean_0.1', 'mean_windx_max_0.4', 'bathy_max_0.05', 
            'mean_windx_max_1.0', 'mean_winds_mean_0.4', 'min_winds_max_0.05', 'mean_windx_min_1.0', 
            'max_pres_max_1.0', 'mean_windy_min_0.4', 'max_windx_max_0.1', 'mean_windy_max_1.0', 
            'max_windy_mean_1.0', 'max_pres_max_0.4', 'mean_windy_min_1.0', 'mean_windy_mean_0.4', 
            'min_windx_min_0.05', 'max_pres_mean_1.0', 'min_windx_min_0.4', 'max_windy_max_0.4', 
            'lons', 'bathy_min_0.4', 'max_pres_mean_0.4', 'mean_windy_min_0.1', 'min_winds_mean_0.4', 
            'mean_winds_max_0.4', 'max_windy_max_0.05', 'min_winds_max_0.1', 'max_windx_min_0.05', 
            'max_winds_max_0.1', 'mean_pres_max_1.0', 'min_windy_min_0.1', 'min_windy_max_0.1', 
            'max_windy_max_0.1', 'min_windx_max_0.1', 'mean_windx_mean_0.1', 'min_winds_mean_0.05', 
            'min_windx_max_1.0', 'mean_windy_min_0.05', 'mean_windx_min_0.4', 'max_pres_max_0.05', 
            'min_windx_mean_0.1', 'max_windx_mean_0.1', 'min_pres_mean_1.0', 'mean_windy_max_0.1', 
            'min_windx_max_0.4', 'bathy_max_0.1', 'min_windy_min_0.4', 'min_windx_mean_1.0', 
            'mean_pres_mean_1.0', 'mean_windy_mean_0.05', 'max_windy_mean_0.1', 'mean_windy_mean_0.1', 
            'mean_windy_max_0.05', 'min_windy_mean_0.05', 'min_windy_mean_1.0', 'mean_pres_min_0.05', 
            'mean_pres_max_0.4', 'min_windy_max_0.05', 'min_windx_mean_0.05', 'min_windx_min_0.1', 
            'min_windy_mean_0.1', 'min_windy_mean_0.4', 'mean_windx_mean_1.0', 'min_windx_min_1.0', 
            'min_windx_mean_0.4', 'min_pres_max_0.1', 'max_windy_mean_0.4', 
            'min_windy_max_1.0', 'min_windy_max_0.4', 'max_windy_min_0.4', 'max_windx_min_1.0', 
            'mean_winds_mean_1.0', 'lats', 'mean_windy_mean_1.0', 'mean_windy_max_0.4']    
         
 

        with h5py.File(fname, 'r') as ds:
            #keys = sorted(list(k for k in ds.keys() if k not in ['zeta_max', 'lat', 'lon', 'bathy', 'mesh_inds']))
            keys = sorted(list(k for k in ds.keys() if k in top_features))
            zeta = ds['zeta_max'][:]

            # Only keep locations where zeta_max > 0
            valid_indices = (zeta > 0.0)
            zeta_filtered = zeta[valid_indices]
     
            mat = np.empty((len(zeta_filtered), len(keys)))
            for i, k in enumerate(keys):
                mat[:, i] = ds[k][:][valid_indices] 

            return {
                'zeta_max': torch.Tensor(zeta_filtered),
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
