import numpy as np 
print("Imported numpy")
import glob
print("Imported glob")
import h5py
print("imported h5py")
import pandas as pd
print("Imported pandas.")
import os
print("Imported os")
from feature_functions import HollandWinds, GridEncoder, save_stats
#from mpi4py import MPI
print("Imported feature functions")
from fire import Fire
print("Imported Fire")

CORRAL_DIR='/corral/projects/NHERI/projects/8647283878534835730-242ac117-0001-012'
BASINS = ["EP", "NA", "NI", "SI", "SP", "WP"]
# have to do this on Frontera because mpi4py currently hangs on import. . . .
rank = int(os.getenv("PMI_RANK", 0))
size = int(os.getenv("PMI_SIZE", 1))
print(f"Rank: {rank}, size: {size}")

class Dataset:
    '''
    class to create dataset in OpenMPI
    '''
    def __init__(self, pr_dir=CORRAL_DIR, downsample_factor=5, window=5,
              output_dir=".", input_format='packed'
            ):
        """Initialize the class
        """
        self.pr_dir = pr_dir
        self.bathy = h5py.File(pr_dir + '/global_bathy.hdf5')['depth'][:]
        mesh_coords = pd.read_csv(pr_dir + '/global_mesh_coords.csv', index_col=0)
        if os.path.exists(pr_dir+"/background_zeta.npz"):
            self.zeta_background = np.load(pr_dir+"/background_zeta.npz")["zeta"]
        else:
            print(f"Warning: Missing background zeta, could not find 'background_zeta.npz' in '{self.pr_dir}'")
            self.zeta_background = None
        self.lats = mesh_coords['lat'].values
        self.lons = mesh_coords['lon'].values
        self.downsample_factor = downsample_factor
        self.window = window
        self.output_dir = output_dir
        self.input_format = input_format
        if input_format not in ['packed', 'unpacked']: raise ValueError(f"Unrecognized input format: {input_format}!")
        print("Initialized class")

    def _mpi_get_data(self):
        """Process dataset in parallel."""
        
        local_storms = self._get_storms()

        for storm in local_storms:
            data = self._get_data(storm)
            save_stats(data, storm['outfile'])

    def _get_data(self, storm):
        
        if self.input_format == "packed":
            dirname = storm['indir']
            #reading the contents of the elevation file
            ele_file = '/elevation.hdf5'
            track_file = dirname+'/track.csv' 

            #read an elevation file
            f = h5py.File(dirname+ele_file, 'r')
            lat_fall = f['landfall_coord'][:][0]
            inds = f['mesh_inds'][:]
            zeta_max = f['zeta_max'][:]
            lon_fall = f['landfall_coord'][:][1]
        elif self.input_format == "unpacked":
            dirname = storm['indir']
            lat_fall, lon_fall = storm['landfall']
            f = np.load(dirname+"/outputs/maxele.npz")
            zeta_max = f["zeta"]
            inds = np.arange(len(zeta_max))
            track_file = storm['track'] 

        if lon_fall > 180:
            lon_fall = lon_fall-360
            
        zeta_max[zeta_max<0] = 0
        trk = pd.read_csv(track_file)
        
        holland = HollandWinds(trk)
        times = np.arange(0, len(trk))
        lats, lons = self.lats[inds], self.lons[inds]
        # we only need a padding of 1 degree around the window for computations
        mask = (np.abs(lats-lat_fall) <= self.window + 1) & (np.abs(lons-lon_fall) <= self.window+1)
        lats, lons, inds, zeta_max = lats[mask], lons[mask], inds[mask], zeta_max[mask]
        coordinates = list(zip(lats, lons))
        
        bathy_fil = self.bathy[inds]
        zeta_diff = zeta_max - self.zeta_background[inds]

        windx = np.zeros((len(times), len(coordinates)))
        windy = np.zeros((len(times), len(coordinates)))
        pres = np.zeros((len(times), len(coordinates)))

        for i,t in enumerate(times):
            wx, wy, p = holland.evaluate(t, lats, lons)
            windx[i, :] = wx
            windy[i, :] = wy
            pres[i, :] = p
        
        features = {'lon': lons, 'lat': lats, 'bathy': bathy_fil, 'zeta_max': zeta_max, 'zeta_diff': zeta_diff}
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

        inds = self._sample_data(features, lat_fall, lon_fall, downsample_factor=self.downsample_factor)
    
        # Extracting the coordinates from the filtered GeoDataFrame
        # Substrings to look for
        substrings = ['pres', 'windx', 'windy', 'lat', 'lon', 'zeta','winds', 'bathy','mesh']

        # Create the dictionary
        selected_data = {}
        N = len(features['zeta_max'])
        assert all(len(features[c]) == N for c in features)
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
                (features['bathy'] > -10)
               )[0]
        N = len(inds)
        downsample_factor = min(N//1000+1, downsample_factor)
        return inds[::downsample_factor]
    
    def _get_storms(self):

        basin, category = self.basin, self.category
        #read all the files
        if self.input_format == "packed":
            dirnames = sorted(glob.glob(f"{self.pr_dir}/{basin}/category{category}/*"))[rank::size]

            saved_directory = f"{self.output_dir}/{basin}/category{category}"
            os.makedirs(saved_directory, exist_ok=True)
            storms = []
            for dirname in dirnames:
              storm_id = dirname.split("/")[-1]
              outfile = saved_directory+"/"+storm_id+".hdf5"
              if os.path.exists(outfile): continue
              storms.append({"indir": dirname, "outfile": outfile})

        elif self.input_format == "unpacked":
            storms = []
            missing = 0
            existing = 0
            print("starting glob.")
            for dirname in sorted(glob.glob(f"{self.pr_dir}/run*/"))[rank::size]:
               print("processing ", dirname)
               landfalls = pd.read_csv(dirname+"/landfalls.csv")
               for i in range(len(landfalls)):
                   row = landfalls.iloc[i]
                   stormdir = dirname+f"/unpacked_inputs/storm{i:02d}/"
                   if not os.path.exists(stormdir+"outputs/maxele.npz"):
                      print("Missing", stormdir+"outputs/maxele.npz")
                      missing += 1
                      continue
                   basin = BASINS[int(row['basin'])]
                   outdir = f"{self.output_dir}/{basin}/category{int(row['cat'])}/"
                   os.makedirs(outdir, exist_ok=True)
                   storm_id = f"{int(row['year'])}{int(row['month']):02d}{int(row['tcnum']):02d}{int(row['tstep']):03d}"
                   outfile = outdir+"/"+storm_id+".hdf5"
                   if os.path.exists(outfile):
                       existing += 1
                       continue
                   storms.append({
                     'indir': stormdir, 'outfile':outfile,
                     'landfall': (row['lat'], row['lon']),
                     'track': dirname+f"/track{i:02d}.csv"})
        
        print("Creating Dataset for {} Storms".format(len(storms)))
        return storms

    def make_background(self, min_files=20):
        """Make background elevation."""
        npzfiles = []
        if self.input_format == "unpacked":
            rundirs = sorted(glob.glob(f"{self.pr_dir}/run*/"))
            for dirname in rundirs:
               landfalls = pd.read_csv(dirname+"/landfalls.csv")
               for i in range(len(landfalls)):
                   row = landfalls.iloc[i]
                   npzfile = dirname+f"/unpacked_inputs/storm{i:02d}/outputs/maxele.npz"
                   if not os.path.exists(npzfile):
                      continue
                   npzfiles.append(npzfile)
               if len(npzfiles) > min_files: break
            
            zetas = []
            for npzfile in npzfiles:
                ark = np.load(npzfile)
                zetas.append(ark["zeta"])

            all_zetas = np.column_stack(zetas)
            all_zetas[all_zetas<0] = 0
            print(all_zetas.shape)
            background_zeta = np.median(all_zetas, axis=1)
            print(background_zeta, background_zeta.shape)
            np.savez(f"{self.pr_dir}/background_zeta.npz", zeta=background_zeta)

        else:
            raise NotImplementedError()  

    def create(self, basin=None, category=None):
        self.basin = basin
        self.category = category
        self._mpi_get_data()

    def check(self, basin, category, storm_id):
        """Check outputs for a single storm
        """
        res = self._get_data(f"{self.pr_dir}/{basin}/category{category}/{storm_id}")
        for k in res:
            if res[k].min() == res[k].max():
                print(k, res[k].min())

if __name__ == "__main__":
    print("Imported libraries, entering script.")
    Fire(Dataset)
