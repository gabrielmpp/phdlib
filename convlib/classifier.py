import xarray as xr
from meteomath import to_cartesian, divergence
import pandas as pd
from convlib.LCS import LCS
import sys
from typing import Optional
import numpy as np
config = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_2000010100-2000123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_2000010100-2000123118.nc',

    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-02-10T18:00:00'),
                   'latitude': slice(5, -35),
                   'longitude': slice(-75, -35)
                    }
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

    def __call__(self, config: dict, method: Optional[str], lcs_type: Optional[str] = None) -> xr.DataArray:
        """

        :rtype: xr.DataArray
        """
        print(f"*---- Calling classifier with method {method} ----*")
        self.method = method
        self.config = config

        u, v = self._read_data
        u = to_cartesian(u)
        v = to_cartesian(v)

        if self.method == 'Q':
            classified_array = self._Q_method(u, v)
        elif self.method == 'conv':
            classified_array = self._conv_method(u, v)
        elif self.method == 'lagrangian':
            assert isinstance(lcs_type, str), 'lcs_type must be string'
            classified_array = self._lagrangian_method(u, v, lcs_type)
        else:
            method = self.method
            raise ValueError(f'Method {method} not supported')
        print("*---- Applying classification method ----*")

        return classified_array

    @property
    def _read_data(self):
        """
        :return: tuple of xarray.dataarray
        """
        print("*---- Reading input data ----*")

        u = xr.open_dataarray(self.config['data_basepath'] + self.config['u_filename'])
        v = xr.open_dataarray(self.config['data_basepath'] + self.config['v_filename'])
        u.coords['longitude'].values = (u.coords['longitude'].values + 180) % 360 - 180
        v.coords['longitude'].values = (v.coords['longitude'].values + 180) % 360 - 180
        u = u.sel(self.config['array_slice'])
        v = v.sel(self.config['array_slice'])
        u = u.resample(time='1D').mean('time')
        v = v.resample(time='1D').mean('time')
        print(u.dims['longitude'])
        new_lon = np.linspace(u.longitude[0].values, u.longitude[-1].values, u.longitude.values.shape * 0.2)
        new_lat = np.linspace(u.latitude[0].values, u.latitude[-1].values, u.longitude.values.shape * 0.2)
        print("*---- Start interp ----*")
        u = u.interp(latitude=new_lat, longitude=new_lon)
        v = v.interp(latitude=new_lat, longitude=new_lon)
        print('*---- Finish interp ----*"')


        print("*---- Done reading ----*")
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
    def _lagrangian_method(u, v, lcs_type: str):
        timestep = pd.infer_freq(u.time.values)
        if 'H' in timestep:
            timestep = float(timestep.replace('H',''))*3600
        elif timestep == 'D':
            timestep = 86400
        else:
            raise ValueError(f"Frequence {timestep} not supported.")

        lcs = LCS(lcs_type=lcs_type)
        eigenvalues = lcs(u, v)/timestep
        return eigenvalues


if __name__ == '__main__':

    classifier = Classifier()

    running_on = str(sys.argv[1])
    #running_on =''

    if running_on == 'jasmin':
        config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
        outpath = '/home/users/gmpp/out/'
    else:
        outpath = 'data/'

    classified_array1 = classifier(config, method='lagrangian', lcs_type='attracting')
    classified_array2 = classifier(config, method='conv')

    classified_array1.to_netcdf(f'{outpath}SL_attracting.nc')

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
