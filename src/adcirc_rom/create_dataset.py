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

class Dataset:
    '''
    class to create dataset in OpenMPI
    '''
    def __init__(self):
        self.bathy = None

    def _mpi_get_data(self, basin, category, pr_dir):
        '''Using OpeMPI to create dataset in Parallel'''
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        local_dirs = self._get_dirs(basin, category, pr_dir)
        #further processing of local directories.

        #reading global bathymetery
        self._get_bathy(pr_dir)

        for dir in local_dirs:
            data = self._get_data(dir, pr_dir)
            output_dir = "." + basin + category + "/" + dir.split("/")[-1]
            save_stats(data, output_dir + ".hdf5") 
    
    def _get_bathy(self, pr_dir):
        self.bathy = h5py.File(pr_dir + '/global_bathy.hdf5')['depth'][:]

    def _get_data(self, dir, pr_dir):
        #reading the contents of the elevation file
        ele_file = '/elevation.hdf5'
        track_file = '/track.csv' 

        #global mesh coordinates
        mesh_coords = pd.read_csv(pr_dir + '/global_mesh_coords.csv', index_col=0)

        downsample_factor = 10

        #read an elevation file
        f = h5py.File(dir+ele_file, 'r')
        lat_fall = f['landfall_coord'][:][0]
        lon_fall = f['landfall_coord'][:][1]

        if lon_fall > 180:
            lon_fall = lon_fall-360
            
        zeta_max = f['zeta_max'][:]
        
        zeta_max[zeta_max<0] = 0
        
        
        trk = pd.read_csv(dir + track_file)
        
        # best_track = pd.read_csv("./track.csv")
        holland = HollandWinds(trk)
        times = np.arange(0, len(trk))
        
        lats = mesh_coords.loc[f['mesh_inds'][:]]['lat']
        lons = mesh_coords.loc[f['mesh_inds'][:]]['lon']
        
        coordinates = list(zip(lats, lons))
        
        bathy_fil = self.bathy[f['mesh_inds'][:]]

        windx = np.zeros((len(times), len(coordinates)))
        windy = np.zeros((len(times), len(coordinates)))
        pres = np.zeros((len(times), len(coordinates)))


        for i,t in enumerate(times):
            wx, wy, p = holland.evaluate(t, lats, lons)
            windx[i, :] = wx
            windy[i, :] = wy
            pres[i, :] = p
        
        # Create GeoDataFrame from your list of points
        gdf_points = pd.DataFrame({
                                        'lats':lats.values,
                                        'lons':lons.values
                                    }, index = np.arange(len(lats)))
        gdf_points['zeta_max'] = zeta_max
        gdf_points['mesh_inds'] = f['mesh_inds'][:]
        
        stats = ['min', 'mean','max']
        
        # Creating a dictionary to map variable names to their values
        variables = {'windx': windx, 'windy': windy, 'pres': pres, 'winds':np.sqrt(windx**2+windy**2), 'bathy':bathy_fil}
        
        for var_name, variable in variables.items():
            
            if var_name == 'bathy':
                
                scales=[5, 10, 40, 100]
                grid_solve = GridEncoder(np.array(lons), np.array(lats))
                stat = grid_solve.encode(var, scales=scales, name=var_name)

                for key, value in stat.items():
                    gdf_points[key] = value
                    
            else:
            
                for st1 in stats: #temporal
                #     for st2 in stats: #spatial
                    variable_record = np.zeros((variable.shape[1]))
                #         for deg in degree:
                    if st1 == 'min':
                        var_abs = np.abs(variable)
                        min_indices = np.argmin(var_abs, axis=0)
                        var = variable[min_indices, np.arange(variable.shape[1])]
                #         wx = np.min(windx_abs, axis = 0)
                    if st1 == 'mean':
                        var = np.mean(variable, axis = 0)
                    if st1 == 'max':
                #         wx = np.max(windx_abs, axis = 0)
                        var_abs = np.abs(variable)
                        max_indices = np.argmax(var_abs, axis=0)
                        var = variable[max_indices, np.arange(variable.shape[1])]
                    scales=[5, 10, 40, 100]
                    grid_solve = GridEncoder(np.array(lons), np.array(lats))
                    stat = grid_solve.encode(var, scales=scales, name=st1+"_"+var_name)

                    for key, value in stat.items():
                        gdf_points[key] = value

        rmax_lanfall = trk[(trk['lat']==lat_fall)&((trk['lon']==lon_fall))].rmax.values
        gdf_points['bathy'] = bathy_fil
        select_nodes = self._sample_data(gdf_points, lat_fall, lon_fall, rmax_lanfall, downsample_factor=10)

        max_surge_point = gdf_points[gdf_points['zeta_max']==gdf_points['zeta_max'].max()]
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
            df = pd.read_csv(file_name)
            # Concatenate the new DataFrame with the existing one
            df = pd.concat([df, df1], axis=0)

            df.to_csv('./record_maxsurge.csv', index=False)
        else:
            # If the file does not exist, create an empty DataFrame with the specified columns
            df1.to_csv('./record_maxsurge.csv', index=False)
    
    
        # Extracting the coordinates from the filtered GeoDataFrame
        # Substrings to look for
        substrings = ['pres', 'windx', 'windy', 'lat', 'lon', 'zeta','winds', 'bathy','mesh']

        # Create the dictionary
        selected_data = {}
        for col in select_nodes.columns:
            if any(substring in col for substring in substrings):
                selected_data[col] = select_nodes[col].tolist()

        return selected_data

    def _windowing(self, gdf_points, center_lat, center_lon, window):
        # Calculate the boundaries
        lat_min = center_lat - window
        lat_max = center_lat + window
        lon_min = center_lon - window
        lon_max = center_lon + window

        # Filtering the points within the square boundary
        points_in_square = gdf_points[
            (gdf_points.lons >= lon_min) & (gdf_points.lons <= lon_max) &
            (gdf_points.lats >= lat_min) & (gdf_points.lats <= lat_max)
        ]
        
        points_in_square = points_in_square[points_in_square['bathy']<10]
    
        return points_in_square
    
    def _sample_data(self, gdf_points, center_lat, center_lon, rmax, downsample_factor):
    
        # Window for filtering
        window = np.ceil(2 * rmax / 111)[0]
        points_in_square = self._windowing(gdf_points, center_lat, center_lon, window)
        
        while len(points_in_square) < 1000:
            window *= 2
            points_in_square = self._windowing(gdf_points, center_lat, center_lon, window)
        
        if len(points_in_square) // downsample_factor >= 1000:
            n_samples = int(len(points_in_square) / downsample_factor)

            # Normalize zeta_max to [0, 1]
            zeta_max_normalized = (points_in_square['zeta_max'] - points_in_square['zeta_max'].min()) / (points_in_square['zeta_max'].max() - points_in_square['zeta_max'].min())

            # Create a Sobol sequence generator for 1-dimensional sampling
            sobol_sampler = qmc.Sobol(d=1)

            # Generate quasi-random samples in the range [0, 1)
            samples = sobol_sampler.random(n=n_samples)

            # Scale the Sobol sequence to match the range of zeta_max
            scaled_samples = qmc.scale(samples, zeta_max_normalized.min(), zeta_max_normalized.max())

            # Map Sobol sequence to zeta_max values
            sampled_zeta_indices = np.searchsorted(zeta_max_normalized.sort_values(), scaled_samples.flatten())

            # Select the rows based on the mapped indices
            sampled_indices = points_in_square.index[sampled_zeta_indices]
            points_in_square = points_in_square.loc[sampled_indices]

        
        elif len(points_in_square) > 1000:
            n_samples = 1000
            
            # Normalize zeta_max to [0, 1]
            zeta_max_normalized = (points_in_square['zeta_max'] - points_in_square['zeta_max'].min()) / (points_in_square['zeta_max'].max() - points_in_square['zeta_max'].min())

            # Create a Sobol sequence generator for 1-dimensional sampling
            sobol_sampler = qmc.Sobol(d=1)

            # Generate quasi-random samples in the range [0, 1)
            samples = sobol_sampler.random(n=n_samples)

            # Scale the Sobol sequence to match the range of zeta_max
            scaled_samples = qmc.scale(samples, zeta_max_normalized.min(), zeta_max_normalized.max())

            # Map Sobol sequence to zeta_max values
            sampled_zeta_indices = np.searchsorted(zeta_max_normalized.sort_values(), scaled_samples.flatten())

            # Select the rows based on the mapped indices
            sampled_indices = points_in_square.index[sampled_zeta_indices]
            points_in_square = points_in_square.loc[sampled_indices]
        
        return points_in_square
    
    def _get_dirs(self, basin, category, pr_dir):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        if rank==0:
            '''
            directory to the project - '/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'
            '''
            # pr_dir = '../../projects/PRJ-4528'
            #read all the files

            NA_files = sorted(glob.glob(pr_dir+basin+category+'/*'))

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


    def create(self, basin, category, pr_dir = '/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'):
        self._mpi_get_data(basin, category, pr_dir)
    
if __name__ == "__main__":
    Fire(Dataset)