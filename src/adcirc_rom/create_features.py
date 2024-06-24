import numpy as np 
import glob
import h5py
import pandas as pd
import math
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine_vector

from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

import matplotlib.pyplot as plt

import os

from feature_functions import HollandWinds, GridEncoder, save_stats
from mpi4py import MPI

#setup for parallel processing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

if rank==0:
    '''
    directory to the project - '/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'
    '''

    # pr_dir = '../../projects/PRJ-4528'

    pr_dir = '/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'

    basin = '/EP'

    category = '/category5'

    #read all the files

    NA_files = sorted(glob.glob(pr_dir+basin+category+'/*'))

    saved_directory = './EP/category5'

    existing_files = sorted(glob.glob(saved_directory+"/*.hdf5"))
    existing_files = [x.split("/")[-1] for x in existing_files]
    existing_files = [x.split(".")[0] for x in existing_files]

    NA_files = [x for x in NA_files if x.split('/')[-1] not in existing_files]

else:
    '''
    directory to the project - '/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'
    '''

    # pr_dir = '../../projects/PRJ-4528'

    pr_dir = '/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'

    basin = '/EP'

    category = '/category5'

    NA_files = None

NA_files = comm.bcast(NA_files, root=root)
NA_files = NA_files[int(len(NA_files)*rank/size):int(len(NA_files)*(rank+1)/size)]

'''
elevation - elevation.hdf5
'''

#reading the contents of the elevation file
ele_file = '/elevation.hdf5'

mesh_coords = pd.read_csv(pr_dir + '/global_mesh_coords.csv', index_col=0)
track_file = '/track.csv'

# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = gpd.read_file('./ne_10m_land/ne_10m_land.shp')

downsample_factor = 10

bathy = h5py.File(pr_dir + '/global_bathy.hdf5')['depth'][:]

for f in NA_files:
    #read a sample elevation file
    f_sample = h5py.File(f+ele_file, 'r')
    lat_fall = f_sample['landfall_coord'][:][0]
    lon_fall = f_sample['landfall_coord'][:][1]

    if lon_fall > 180:
        lon_fall = lon_fall-360
        
    zeta_max = f_sample['zeta_max'][:]
    
    zeta_max[zeta_max<0] = 0
    
    
    trk = pd.read_csv(f + track_file)
    
    # best_track = pd.read_csv("./track.csv")
    holland = HollandWinds(trk)
    times = np.arange(0, len(trk))
    
    lats = mesh_coords.loc[f_sample['mesh_inds'][:]]['lat']
    lons = mesh_coords.loc[f_sample['mesh_inds'][:]]['lon']
    
    coordinates = list(zip(lats, lons))
    
    bathy_fil = bathy[f_sample['mesh_inds'][:]]
    
#     windx = np.zeros((len(times), len(filtered_coordinates)))
#     windy = np.zeros((len(times), len(filtered_coordinates)))
#     pres = np.zeros((len(times), len(filtered_coordinates)))
    
    windx = np.zeros((len(times), len(coordinates)))
    windy = np.zeros((len(times), len(coordinates)))
    pres = np.zeros((len(times), len(coordinates)))
    
#     lats = [x[0] for x in filtered_coordinates]
#     lons = [x[1] for x in filtered_coordinates]
    
    for i,t in enumerate(times):
        wx, wy, p = holland.evaluate(t, lats, lons)
        windx[i, :] = wx
        windy[i, :] = wy
        pres[i, :] = p
        
    # Create GeoDataFrame from your list of points
    gdf_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in coordinates])
    gdf_points['zeta_max'] = zeta_max
    gdf_points['mesh_inds'] = f_sample['mesh_inds'][:]
    
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

    
    
    gdf_points.crs = world.crs
    
    center_lat, center_lon = lat_fall, lon_fall  # Replace lat1, lon1 with your actual center coordinates

    # Calculate the boundaries
    lat_min = center_lat - 1
    lat_max = center_lat + 1
    lon_min = center_lon - 1
    lon_max = center_lon + 1

        # Filtering the points within the square boundary
    points_in_square = gdf_points[
        (gdf_points.geometry.x >= lon_min) & (gdf_points.geometry.x <= lon_max) &
        (gdf_points.geometry.y >= lat_min) & (gdf_points.geometry.y <= lat_max)]

    # Perform a spatial join between points and land polygons
    points_on_land = gpd.sjoin(points_in_square, world, how="inner", op='intersects')



    rand_nodes = np.random.permutation(len(points_on_land))


    select_nodes = points_on_land.iloc[rand_nodes[:int(len(rand_nodes)/downsample_factor)]]
    
     # Extracting the coordinates from the filtered GeoDataFrame
#     filtered_coordinates = [(point.y, point.x) for point in select_nodes.geometry]
    # Substrings to look for
    substrings = ['pres', 'windx', 'windy', 'lats', 'lons', 'zeta','winds', 'bathy']

    # Create the dictionary
    selected_data = {}
    for col in select_nodes.columns:
        if any(substring in col for substring in substrings):
            selected_data[col] = select_nodes[col].tolist()
    
    
    output_dir = "./EP/category5/" + f.split("/")[-1]
    
    save_stats(selected_data, output_dir + ".hdf5")
    del stat, gdf_points, select_nodes, points_in_square


MPI.Finalize()