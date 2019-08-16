import xarray as xr
from meteomath import to_cartesian, divergence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from convlib.LCS import LCS
import sys

config = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_2000010100-2000123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_2000010100-2000123118.nc'
    }

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

    @staticmethod
    def _lagrangian_method(u, v):
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


if __name__ == '__main__':

    classifier = Classifier()

    running_on = str(sys.argv[1])

    if running_on == 'jasmin':
        config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
        outpath = '/home/users/gmpp/'
    else:
        outpath = 'data/'

    classified_array1 = classifier(config, method ='lagrangian')
    classified_array2 = classifier(config, method = 'conv')

    classified_array1.to_netcdf(f'{outpath}SL.nc')

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
