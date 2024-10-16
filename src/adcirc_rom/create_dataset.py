import numpy as np 
import glob
import h5py
import pandas as pd
import math
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine_vector, haversine
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
import matplotlib.pyplot as plt
import os
from scipy.stats import qmc
from feature_functions import HollandWinds, GridEncoder, save_stats
from mpi4py import MPI
from fire import Fire

CORRAL_DIR='/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'

class Dataset:
    '''
    class to create dataset in OpenMPI
    '''
    def __init__(self, pr_dir=CORRAL_DIR, downsample_factor=5, window=5):
        """Initialize the class
        """
        self.pr_dir = pr_dir
        self.bathy = h5py.File(pr_dir + '/global_bathy.hdf5')['depth'][:]
        mesh_coords = pd.read_csv(pr_dir + '/global_mesh_coords.csv', index_col=0)
        self.lats = mesh_coords['lat'].values
        self.lons = mesh_coords['lon'].values
        self.downsample_factor = downsample_factor
        self.window = window

    def _mpi_get_data(self, basin, category):
        '''Using OpeMPI to create dataset in Parallel'''
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        local_dirs = self._get_dirs(basin, category)
        #further processing of local directories.

        #reading global bathymetery
        self._get_bathy(pr_dir)

        for dirname in local_dirs:
            data = self._get_data(dirname)
            output_dir = "." + basin + category + "/" + dirname.split("/")[-1]
            save_stats(data, output_dir + ".hdf5")
    
    def _get_data(self, dirname):
        #reading the contents of the elevation file
        ele_file = '/elevation.hdf5'
        track_file = '/track.csv' 

        #read an elevation file
        f = h5py.File(dirname+ele_file, 'r')
        lat_fall = f['landfall_coord'][:][0]
        lon_fall = f['landfall_coord'][:][1]

        if lon_fall > 180:
            lon_fall = lon_fall-360
            
        zeta_max = f['zeta_max'][:]
        zeta_max[zeta_max<0] = 0
        trk = pd.read_csv(dirname + track_file)
        
        holland = HollandWinds(trk)
        times = np.arange(0, len(trk))
        inds = f['mesh_inds'][:]
        lats, lons = self.lats[inds], self.lons[inds]
        # we only need a padding of 1 degree around the window for computations
        mask = (np.abs(lats-lat_fall) <= self.window + 1) & (np.abs(lons-lon_fall) <= self.window+1)
        lats, lons, inds = lats[mask], lons[mask], inds[mask]
        coordinates = list(zip(lats, lons))
        
        bathy_fil = self.bathy[inds]

        windx = np.zeros((len(times), len(coordinates)))
        windy = np.zeros((len(times), len(coordinates)))
        pres = np.zeros((len(times), len(coordinates)))

        for i,t in enumerate(times):
            wx, wy, p = holland.evaluate(t, lats, lons)
            windx[i, :] = wx
            windy[i, :] = wy
            pres[i, :] = p
        
        features = {'lon': lons, 'lat': lats, 'bathy': bathy_fil}
        stats = ['min', 'mean','max']
        encoder = GridEncoder(lons, lats) 
        # Creating a dictionary to map variable names to their values
        variables = {'windx': windx, 'windy': windy, 'pres': pres, 'winds':np.sqrt(windx**2+windy**2), 'bathy':bathy_fil}
        
        for var_name, variable in variables.items():
            if var_name == 'bathy':
                
                scales=[5, 10, 40, 100]
                features.update(encoder.encode(variable, scales=scales, name=var_name))
            else:
            
                for st1 in stats: #temporal
                    if st1 == 'min':
                        min_indices = np.argmin(variable, axis=0)
                        var = variable[min_indices, np.arange(variable.shape[1])]
                    if st1 == 'mean':
                        var = np.mean(variable, axis = 0)
                    if st1 == 'max':
                        max_indices = np.argmax(variable, axis=0)
                        var = variable[max_indices, np.arange(variable.shape[1])]
                    scales=[5, 10, 40, 100]
                    features.update(encoder.encode(var, scales=scales, name=st1+"_"+var_name))

        #rmax_lanfall = trk[(trk['lat']==lat_fall)&((trk['lon']==lon_fall))].rmax.values
        inds = self._sample_data(features, lat_fall, lon_fall, downsample_factor=self.downsample_factor)
        print(f"Reduced from {len(lats)} to {len(inds)}")
        """
        max_surge_point = np.argmax(features['zeta_max'])
        # Check if the file exists
        file_name = 'record_maxsurge.csv'
        file_exists = os.path.isfile(file_name)

        # Define the data for the new DataFrame
        data = {
            'Hurricane': [dir.split('/')[-1]],
            'Basin': [dir.split('/')[-3]],
            'Category': [dir.split('/')[-2]],
            'landfall_lat': [lat_fall],
            'landfall_lon': [lon_fall],
            'Max_Surge_recorded':[select_nodes.zeta_max.max()],
            'Max_Surge': [max_surge_point.zeta_max.values[0]], 
            'Max_surge_included': [max_surge_point.index.isin(select_nodes.index)[0]],
            'Max_Surge_lat': [max_surge_point.lats.values[0]],
            'Max_Surge_lon': [max_surge_point.lons.values[0]], 
            'Landfall_to_MaxSurge': [haversine((lat_fall, lon_fall), (max_surge_point.lats.values[0], max_surge_point.lons.values[0]))]
        }

        # Define the index
        index = [dir.split('/')[-1]]

        # Create the new DataFrame
        df1 = pd.DataFrame(index=index, data=data)

        

        if file_exists:
            # If the file exists, open it
            df = pd.read_csv(file_name, na_values=[], keep_default_na=False)
            # Concatenate the new DataFrame with the existing one
            df = pd.concat([df, df1], axis=0)

            df.to_csv('./record_maxsurge.csv', index=False)
        else:
            # If the file does not exist, create an empty DataFrame with the specified columns
            df1.to_csv('./record_maxsurge.csv', index=False)
        """
    
        # Extracting the coordinates from the filtered GeoDataFrame
        # Substrings to look for
        substrings = ['pres', 'windx', 'windy', 'lat', 'lon', 'zeta','winds', 'bathy','mesh']

        # Create the dictionary
        selected_data = {}
        for col in features:
            if any(substring in col for substring in substrings):
                selected_data[col] = features[col][inds]

        return selected_data

    def _sample_data(self, features, center_lat, center_lon, downsample_factor):
        """Select a subsample of the points
        """
        # Window for filtering
        window = self.window
        inds = np.where(
                (features['lon'] <= center_lon+window) &
                (features['lon'] >= center_lon-window) &
                (features['lat'] <= center_lat+window) &
                (features['lat'] >= center_lat-window) &
                (features['bathy'] < 10)
               )[0]
        N = len(inds)
        downsample_factor = min(N//1000+1, downsample_factor)
        return inds[::downsample_factor]
    
    def _get_dirs(self, basin, category):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        if rank==0:
            #read all the files

            NA_files = sorted(glob.glob(f"{self.pr_dir}/{basin}/category{category}/*"))

            saved_directory = '.'+basin+category

            existing_files = sorted(glob.glob(saved_directory+"/*.hdf5"))
            existing_files = [x.split("/")[-1] for x in existing_files]
            existing_files = [x.split(".")[0] for x in existing_files]

            NA_files = [x for x in NA_files if x.split('/')[-1] not in existing_files]

            print("Creating Dataset for {} Files".format(len(NA_files)))
        else:
            NA_files = None

        NA_files = comm.bcast(NA_files, root=root)
        NA_files = NA_files[int(len(NA_files)*rank/size):int(len(NA_files)*(rank+1)/size)]
        return NA_files


    def create(self, basin, category):
        self._mpi_get_data(basin, category)

    def check(self, basin, category, storm_id):
        """Check outputs for a single storm
        """
        res = self._get_data(f"{self.pr_dir}/{basin}/category{category}/{storm_id}")
        for k in res:
            if res[k].min() == res[k].max():
                print(k, res[k].min())

if __name__ == "__main__":
    Fire(Dataset)
