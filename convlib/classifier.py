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

    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-03-01T18:00:00'),
                   'latitude': slice(15, -50),
                   'longitude': slice(-100, -5)
                   #'latitude': slice(-30, -45),
                   #'longitude': slice(-40, -25)
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

    def __call__(self, config: dict, method: str, lcs_type: Optional[str] = None) -> xr.DataArray:
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
        new_lon = np.linspace(u.longitude[0].values, u.longitude[-1].values, int(u.longitude.values.shape[0] * 0.5))
        new_lat = np.linspace(u.latitude[0].values, u.latitude[-1].values, int(u.longitude.values.shape[0] * 0.5))
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

        timestep_u = pd.infer_freq(u.time.values)
        timestep_v = pd.infer_freq(v.time.values)
        assert timestep_u == timestep_v, "u and v timesteps are different!"
        timestep = timestep_u
        if 'H' in timestep:
            timestep = float(timestep.replace('H', '')) * 3600
        elif timestep == 'D':
            timestep = 86400
        else:
            raise ValueError(f"Frequency {timestep} not supported.")
        u.name = 'u'
        v.name = 'v'
        ds = xr.merge([u, v])
        print(ds)
        lcs = LCS(lcs_type=lcs_type, timestep=timestep, dataarray_template=u.isel(time=0).drop('time'))
        eigenvalues = xr.apply_ufunc(lambda x, y: lcs(u=x, v=y), u, v, dask='parallelized')
        #eigenvalues = ds.groupby('time').apply(lcs)
        return eigenvalues


if __name__ == '__main__':

    classifier = Classifier()

    running_on = str(sys.argv[1])
    lcs_type = str(sys.argv[2])
    year = str(sys.argv[3])
    config['array_slice']['time'] = slice(f'{year}-01-01T00:00:00', f'{year}-12-31T18:00:00')
    #running_on =''
    #lcs_type = 'attracting'
    if running_on == 'jasmin':
        config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
        outpath = '/group_workspaces/jasmin4/upscale/gmpp_convzones/'
    else:
        outpath = 'data/'

    classified_array1 = classifier(config, method='lagrangian', lcs_type=lcs_type)

    classified_array1.to_netcdf(f'{outpath}SL_{lcs_type}_{year}.nc')

