import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import datashader as ds
from datashader import transfer_functions as tf
from datashader.mpl_ext import dsshow
from pyproj import Transformer
import plotly.graph_objects as go
import os


class Visualization:
    """A visualization script for plotting maximum water elevation
    and storm track for a single storm
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mapbox_token = "pk.eyJ1IjoiYnBhY2hldiIsImEiOiJja3kzeGs1ZnMwNmp1MnByMHdsMDNxc241In0.OjVGa4NqNxKxy221aLSAtA"
        self.gcs_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)

    def find_directory(self, storm_number):
        for root, dirs, files in os.walk(self.base_dir):
            if storm_number in dirs:
                return os.path.join(root, storm_number)
        return None

    def load_elevation_data(self, elevation_file):
        data = {}
        with h5py.File(elevation_file, 'r') as file:
            for key in file.keys():
                if file[key].shape == ():  
                    data[key] = file[key][()]
                else:
                    data[key] = file[key][:]
        return data

    def plot_spatial_surge(self, data, track, width=1000, height=500, lat_range=None, lon_range=None):
        df = pd.DataFrame({
            "lon": data["lon"],
            "lat": data["lat"],
            "zeta_max": data["zeta_max"]
        })
        if lat_range:
            df = df[(df['lat'] <= lat_range[1]) & (df['lat'] >= lat_range[0])]
        if lon_range:
            df = df[(df['lon'] <= lon_range[1]) & (df['lon'] >= lon_range[0])]
        
        cvs = ds.Canvas(plot_width=width, plot_height=height)
        df['lon2'], df['lat2'] = self.gcs_to_3857.transform(df['lon'], df['lat'])
        agg = cvs.points(df[['lat2', 'lon2', 'zeta_max']], "lon2", "lat2", ds.mean("zeta_max"))
        cmap = cm.turbo
        img = tf.shade(agg, cmap, how='linear')
        img = img.to_pil()
        coords_lat, coords_lon = sorted(df['lat'].values), sorted(df['lon'].values)
        coordinates = [[coords_lon[0], coords_lat[0]],
                       [coords_lon[-1], coords_lat[0]],
                       [coords_lon[-1], coords_lat[-1]],
                       [coords_lon[0], coords_lat[-1]]]
        
        layout = go.Layout(
            title="Max Surge Heatmap and Storm Track",
            height=600,
            width=800,
            margin=dict(t=80, b=0, l=0, r=0),
            font=dict(color='dark grey', size=18),
            showlegend=True,
            legend=dict(font=dict(size=12)),
            mapbox=dict(
                accesstoken=self.mapbox_token,
                center=dict(
                    lat=df['lat'].mean(),
                    lon=df['lon'].mean(),
                ),
                zoom=5.5,
                style="basic"
            )
        )
        elevation_trace = go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode='markers',
            marker=dict(
                color=df["zeta_max"],
                colorbar=dict(len=.95,
                              y=.45,
                              title=dict(text="Max Surge (m)",
                                         font=dict(size=12))),
                colorscale='Turbo',  
                showscale=True,
                cmin=0,
                cmax=df['zeta_max'].max(),
                size=5)
        )
        elevation_trace.name = "Max Surge"
        
        track_trace = go.Scattermapbox(
            lat=track["lat"],
            lon=track["lon"],
            mode='lines',
            line=dict(width=6, color='black'),
            name='Storm Track'
        )
        
        figure = go.Figure(data=[elevation_trace, track_trace])
        figure.update_layout(layout)
        figure.update_layout(mapbox_style="basic",
                             mapbox_accesstoken=self.mapbox_token,
                             mapbox_layers=[
                                 {
                                     "sourcetype": "image",
                                     "source": img,
                                     "coordinates": coordinates[::-1]
                                 },
                             ],
                             margin={"t": 80, "l": 0, "r": 0, "b": 0}
                             )
        figure.show()

    def plot_storm_surge(self, storm_number):
        storm_dir = self.find_directory(storm_number)
        if storm_dir is None:
            print(f"Storm number {storm_number} not found.")
            return
        
        #Load the surge data
        elevation_file = os.path.join(storm_dir, 'elevation.hdf5')
        elevation_data = self.load_elevation_data(elevation_file)

        #Load mesh coord
        mesh_coords_file = os.path.join(self.base_dir, 'global_mesh_coords.csv')
        mesh_coords_data = pd.read_csv(mesh_coords_file)

        #Combine data 
        mesh_inds = elevation_data['mesh_inds']
        zeta_max = elevation_data['zeta_max']
        combined_data = pd.DataFrame({
            'lat': mesh_coords_data.loc[mesh_inds, 'lat'].values,
            'lon': mesh_coords_data.loc[mesh_inds, 'lon'].values,
            'zeta_max': zeta_max
        })

        #Load storm track csv data
        track_file = os.path.join(storm_dir, 'track.csv')
        track_data = pd.read_csv(track_file)

        # Plot
        self.plot_spatial_surge(combined_data, track_data)
