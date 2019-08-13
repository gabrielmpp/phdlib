import xarray as xr
from meteomath import to_cartesian, divergence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from skimage.feature import blob_dog, blob_log, blob_doh
import sys

config = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_2000010100-2000123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_2000010100-2000123118.nc'
    }


class LCS:
    '''
    Methods to compute LCS in 2D wind fields in xarrray dataarrays
    '''

    def __init__(self):
        pass
    def __call__(self, u, v):
        '''

        :param u: xarray datarray containing u-wind component
        :param v: xarray dataarray containing v-wind component
        :return: xarray dataarray containg
        '''

        def_tensor = self._compute_deformation_tensor(u, v)
        def_tensor = def_tensor.stack({'points': ['time', 'latitude', 'longitude']})
        def_tensor = def_tensor.dropna(dim='points')
        eigenvalues = xr.apply_ufunc(lambda x: self._compute_eigenvalues(x), def_tensor.groupby('points'))
        eigenvalues = eigenvalues.unstack('points').isel(derivatives=0) # isel is a cheap fix for poor array construction in _compute deformation tensor
        return eigenvalues

    @staticmethod
    def _compute_eigenvalues(def_tensor):
        d_matrix = def_tensor.reshape([2,2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)
        eigenvalues = max(np.linalg.eig(cauchy_green.reshape([2,2]))[0]) # Taking the minimum eigenvalue (lambda-1) where the minima correspond to shrinklines
        #eigenvalues = np.log(np.sqrt(eigenvalues))
        eigenvalues = np.repeat(eigenvalues,4).reshape([4,1]) # repeating the same value 4 times just to fill the array, see comment in function call
        return eigenvalues

    @staticmethod
    def _compute_deformation_tensor(u ,v):
        timestep_u = pd.infer_freq(u.time.values)
        timestep_v = pd.infer_freq(v.time.values)
        assert timestep_u == timestep_v, "u and v timesteps are different!"
        timestep = timestep_u

        if 'H' in timestep:
            timestep = float(timestep.replace('H',''))*3600
        elif timestep == 'D':
            timestep = 86400
        else:
            raise ValueError(f"Frequence {timestep} not supported.")

        x_futur = u.x + u*timestep
        y_futur = u.y + v*timestep

        # Solid boundary conditions TODO: improve to more realistic case
        #x_futur = x_futur.where(x_futur > u.x.min(), u.x.min()).where(x_futur < u.x.max(), u.x.max())
        #y_futur = y_futur.where(y_futur > u.y.min(), u.y.min()).where(y_futur < u.y.max(), u.y.max())

        dxdx = x_futur.differentiate('x')
        dxdy = x_futur.differentiate('y')
        dydy = y_futur.differentiate('y')
        dydx = y_futur.differentiate('x')

        dxdx = dxdx.transpose('time','latitude','longitude')
        dxdy = dxdy.transpose('time','latitude','longitude')
        dydy = dydy.transpose('time','latitude','longitude')
        dydx = dydx.transpose('time','latitude','longitude')
        def_tensor=xr.concat([dxdx,dxdy,dydx,dydy],dim=pd.Index(['dxdx', 'dxdy', 'dydx', 'dydy'], name='derivatives'))
        def_tensor = def_tensor.transpose('time', 'derivatives', 'latitude', 'longitude')

        return def_tensor


class Classifier:
    """
    Convergence zones classifier
    """

    def __init__(self):
        '''

        :param config:  config dict
        :param method: classification method. options are: Q
        '''
        pass

    def __call__(self, config: dict, method = 'Q'):
        self.method = method
        self.config = config

        u, v = self._read_data
        u = to_cartesian(u)
        v = to_cartesian(v)

        if self.method == 'Q':
            _classification_method = self._Q_method
        elif self.method == 'conv':
            _classification_method = self._conv_method
        elif self.method == 'lagrangian':
            _classification_method = self._lagrangian_method
        else:
            method = self.method
            raise ValueError(f'Method {method} not supported')

        classified_array = _classification_method(u, v)

        return classified_array

    @property
    def _read_data(self):
        """
        :return: tuple of xarray.dataarray
        """
        u = xr.open_dataarray(config['data_basepath']+ self.config['u_filename'])
        v = xr.open_dataarray(self.config['data_basepath']+ self.config['v_filename'])
        # dia 06-feb 15 feb
        u.coords['longitude'].values = (u.coords['longitude'].values + 180) % 360 - 180
        v.coords['longitude'].values = (v.coords['longitude'].values + 180) % 360 - 180
        u = u.sel(time=slice('2000-02-06T00:00:00', '2000-07-16T18:00:00'), latitude=slice(5, -35),
                  longitude=slice(-75, -35))
        v = v.sel(time=slice('2000-02-06T00:00:00', '2000-02-16T18:00:00'), latitude=slice(5, -35),
                  longitude=slice(-75, -35))
        u = u.resample(time='1D').mean('time')
        v = v.resample(time='1D').mean('time')
        return u, v

    @staticmethod
    def _Q_method(u, v):
        classified_array = 2*u.differentiate('x')*v.differentiate('y') - 2*v.differentiate("x") * u.differentiate("y")
        #classified_array = classified_array.where(Q < 0, 0)
        return classified_array

    @staticmethod
    def _conv_method(u ,v):
        div = divergence(u, v)
       # classified_array = div.where(div<-0.5e-3)
        return div

    def _lagrangian_method(self, u, v):
        timestep = pd.infer_freq(u.time.values)
        if 'H' in timestep:
            timestep = float(timestep.replace('H',''))*3600
        elif timestep == 'D':
            timestep = 86400
        else:
            raise ValueError(f"Frequence {timestep} not supported.")

        lcs = LCS()
        eigenvalues = lcs(u, v)/timestep
        return eigenvalues


class Tracker:

    def __init__(self):
        from skimage.feature import blob_dog, blob_log, blob_doh
        pass

    def _identify_features(self, array):
        array = (array - array.min()) / array.max()
        for time in array.time.values:
            array2D = array.sel(time=time).values
            array2D.shape()
            blobs_doh = blob_doh(array2D, max_sigma=30, threshold=.01)
            print(';a')


        pass

    def track(self, array):
        identified_array = self._identify_features(array)


class Normalizer():
    '''
    Normalizer for xarray datasets
    '''

    def __init__(self, alongwith):
        self._alongwith = alongwith

    def fit(self, X, y=None):
        '''
        :ds: xarray  dataset
        :alongwith: list of sample dimensions
        '''
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')

        X = X.stack({'alongwith': self._alongwith})
        self._mean = X.mean('alongwith')
        self._stdv = X.var('alongwith')**0.5
        return self

    def transform(self, X):
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
        X = X.stack({'alongwith': self._alongwith})
        X = (X - self._mean)/self._stdv
        return X.unstack('alongwith')

    def inverse_transform(self, X):
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
        X = X.stack({'alongwith': self._alongwith})
        X = X * self._stdv + self._mean

        return X.unstack('alongwith')


if __name__ == '__main__':

    classifier = Classifier()

    running_on = str(sys.argv[1])

    if running_on == 'jasmin':
        config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'

    classified_array1 = classifier(config, method ='lagrangian')
    #classified_array2 = classifier(config, method = 'conv')

    classified_array1.to_netcdf('/home/users/gmpp/SL.nc')

    '''
    ntimes=10
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    for time in np.arange(ntimes):
        f, axarr = plt.subplots(1, 2, figsize=(20,10),subplot_kw={'projection': ccrs.PlateCarree()})

        classified_array1.isel(time=time+1).plot(ax=axarr[0],cmap='nipy_spectral', transform=ccrs.PlateCarree())
        #classified_array1.isel(time=time+2).plot( ax=axarr[1], transform=ccrs.PlateCarree())

        #classified_array1.isel(time=time+1).plot( ax=axarr[0],cmap='nipy_spectral',vmin=0,vmax=40, transform = ccrs.PlateCarree())

        classified_array2.isel(time=time+1).plot(vmin = -1e-3,vmax=1e-3,cmap='RdBu', ax=axarr[1], transform = ccrs.PlateCarree())
        axarr[0].add_feature(states_provinces, edgecolor='gray')
        axarr[1].add_feature(states_provinces, edgecolor='gray')
        axarr[0].coastlines()
        axarr[1].coastlines()
        plt.savefig(f'tempfigs/fig{time}.png')
        plt.close()

    #Tracker().track(classified_array1)
    '''
