import xarray as xr
from meteomath import to_cartesian, divergence
import matplotlib.pyplot as plt
import pandas as Pd
import numpy as np
import pandas as pd
from skimage.feature import blob_dog, blob_log, blob_doh

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
        cauchy_green = d_matrix.T * d_matrix
        eigenvalues = np.linalg.eig(cauchy_green.reshape([2,2]))[0][1]#first index take eigenvalue (or vector) second index take the first or second eigenvalue
        eigenvalues = np.repeat(eigenvalues,4).reshape([4,1])
        return eigenvalues
    @staticmethod
    def _compute_deformation_tensor(u ,v):
        timestep = pd.infer_freq(u.time.values)
        x_res = pd.infer_freq(u.x.values)
        if 'H' in timestep:
            timestep = float(timestep.replace('H',''))*3600
        else:
            raise ValueError(f"Frequence {timestep} not supported.")

        x_futur = u.x + u*timestep
        y_futur = u.y + v*timestep

        # Solid boundary conditions
        x_futur = x_futur.where(x_futur > u.x.min(), u.x.min()).where(x_futur < u.x.max(), u.x.max())
        y_futur = y_futur.where(y_futur > u.y.min(), u.y.min()).where(y_futur < u.y.max(), u.y.max())

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



class Classifier():
    """
    Convergence zones classifier
    """

    def __init__(self):
        '''

        :param config:  config dict
        :param method: classification method. options are: Q
        '''
        pass

    def __call__(self, config, method = 'Q'):
        self.method = method
        self.config = config

        u, v = self._read_data
        u = to_cartesian(u)
        v = to_cartesian(v)

        if self.method == 'Q':
            _classification_method = self._Q_method
        if self.method == 'conv':
            _classification_method = self._conv_method
        if self.method == 'lagrangian':
            _classification_method = self._lagrangian_method(u, v)

        classified_array = _classification_method(u, v)

        '''
        f, axarr = plt.subplots(2)
        classified_array1.isel(time=0).plot(vmin=-1e-6, vmax=1e-6, cmap='RdBu', ax=axarr[0])
        classified_array2.isel(time=0).plot(vmin=-1e-3, vmax=1e-3, cmap='RdBu', ax=axarr[1])
        '''

        return classified_array

    @property
    def _read_data(self):
        """
        :return: tuple of xarray.dataarray
        """
        u = xr.open_dataarray(config['data_basepath']+ self.config['u_filename'])
        v = xr.open_dataarray(self.config['data_basepath']+ self.config['v_filename'])
        # dia 06-feb 15 feb
        u = u.sel(time=slice('2000-02-07T00:00:00', '2000-07-07T18:00:00'), latitude=slice(-15,-25), longitude=slice(330,340))
        v = v.sel(time=slice('2000-02-07T00:00:00', '2000-02-07T18:00:00'), latitude=slice(-15,-25), longitude=slice(330,340))
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
        lcs = LCS()
        eigenvalues = lcs(u, v)
        print(eigenvalues)


class Tracker:

    def __init__(self):
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
    classified_array1 = classifier(config, method ='lagrangian')
    classified_array2 = classifier(config, method = 'Q')


    ntimes=40
    import cartopy.crs as ccrs

    for time in np.arange(ntimes):
        f, axarr = plt.subplots(1, 2, figsize=(20,10),subplot_kw={'projection': ccrs.PlateCarree()})

        classified_array1.isel(time=time+1).plot(vmin = -1e-3,vmax=1e-3,cmap='RdBu', ax=axarr[0], transform = ccrs.PlateCarree())
        classified_array2.isel(time=time+1).plot(vmin = -1e-6,vmax=1e-6,cmap='RdBu', ax=axarr[1], transform = ccrs.PlateCarree())
        #axarr[0].coastlines()
        #axarr[1].coastlines()
        plt.savefig(f'tempfigs/fig{time}.png')
        plt.close()

    Tracker().track(classified_array1)
