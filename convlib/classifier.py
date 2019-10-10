import xarray as xr
from meteomath import to_cartesian, divergence
import pandas as pd
from convlib.LCS import LCS
import sys
from typing import Optional
import numpy as np
import concurrent.futures
from convlib.xr_tools import get_seq_mask


config = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_2000010100-2000123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_2000010100-2000123118.nc',
    'tcwv_filename': 'tcwv_ERA5_6hr_2000010100-2000123118.nc',
    'time_freq': '6H',
    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-02-07T18:00:00'),
                   'latitude': slice(-15, -50),
                   'longitude': slice(-90, -5),
                   # 'latitude': slice(-20, -35),
                   # 'longitude': slice(-55, -35)
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

    def __call__(self, config: dict, method: str, lcs_type: Optional[str] = None, lcs_time_len: Optional[int] = 4) -> xr.DataArray:
        """

        :rtype: xr.DataArray
        """
        print(f"*---- Calling classifier with method {method} ----*")
        self.method = method
        self.config = config

        u, v = self._read_data
        u = to_cartesian(u)
        v = to_cartesian(v)
        print("*---- Applying classification method ----*")

        if self.method == 'Q':
            classified_array = self._Q_method(u, v)
        elif self.method == 'conv':
            classified_array = self._conv_method(u, v)
        elif self.method == 'lagrangian':
            assert isinstance(lcs_type, str), 'lcs_type must be string'
            classified_array = self._lagrangian_method(u, v, lcs_type, lcs_time_len)
        else:
            method = self.method
            raise ValueError(f'Method {method} not supported')

        return classified_array

    @property
    def _read_data(self):
        """
        :return: tuple of xarray.dataarray
        """
        print("*---- Reading input data ----*")
        ########### TODO WARNING SELECTING LIMITED NUMB OF DAYS
        u = xr.open_dataarray(self.config['data_basepath'] + self.config['u_filename'])
        v = xr.open_dataarray(self.config['data_basepath'] + self.config['v_filename'])
        tcwv = xr.open_dataarray(self.config['data_basepath'] + self.config['tcwv_filename'])

        u.coords['longitude'].values = (u.coords['longitude'].values + 180) % 360 - 180
        v.coords['longitude'].values = (v.coords['longitude'].values + 180) % 360 - 180
        tcwv.coords['longitude'].values = (tcwv.coords['longitude'].values + 180) % 360 - 180

        u = u.sel(self.config['array_slice'])
        v = v.sel(self.config['array_slice'])
        tcwv = tcwv.sel(self.config['array_slice'])
        # tcwv = tcwv.where(tcwv > 10, 10)


        assert pd.infer_freq(u.time.values) == pd.infer_freq(v.time.values), "u and v should have equal time frequencies"
        data_time_freq = pd.infer_freq(u.time.values)
        if data_time_freq != self.config['time_freq']:
            print("Resampling data to {}".format(self.config['time_freq']))
            u = u.resample(time=self.config['time_freq']).interpolate('linear')
            v = v.resample(time=self.config['time_freq']).interpolate('linear')
            tcwv = tcwv.resample(time=self.config['time_freq']).interpolate('linear')

        if 'viwv' in self.config['u_filename']:
            print("Applying unit conversion")
            u = u / tcwv.values
            v = v / tcwv.values
            print("Done unit conversion")


        new_lon = np.linspace(u.longitude[0].values, u.longitude[-1].values, int(u.longitude.values.shape[0] * 0.5))
        new_lat = np.linspace(u.latitude[0].values, u.latitude[-1].values, int(u.longitude.values.shape[0] * 0.5))
        print("*---- NOT Start interp ----*")
        # u = u.interp(latitude=new_lat, longitude=new_lon)
        # v = v.interp(latitude=new_lat, longitude=new_lon)
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

    def _lagrangian_method(self, u, v, lcs_type: str, lcs_time_len=4) -> xr.DataArray:
        parallel = True
        time_dir = 'backward'
        u = get_seq_mask(u, 'time', lcs_time_len)
        v = get_seq_mask(v, 'time', lcs_time_len)
        shearless = False


        timestep = self.config['time_freq']
        if 'H' in timestep:
            timestep = float(timestep.replace('H', '')) * 3600
        elif 'D' in timestep:
            timestep = float(timestep.replace('D', '')) * 86400
        else:
            raise ValueError(f"Frequency {timestep} not supported.")

        timestep = -timestep if time_dir == 'backward' else timestep

        u.name = 'u'
        v.name = 'v'
        ds = xr.merge([u, v])
        ds_groups = list(ds.groupby('seq'))
        input_arrays = []
        for label, group in ds_groups: # have to do that because bloody groupby returns the labels
            input_arrays.append(group)

        lcs = LCS(lcs_type=lcs_type, timestep=timestep, timedim='time', shearless=shearless)
        array_list = []
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                for i, resulting_array in enumerate(executor.map(lcs, input_arrays)):
                    array_list.append(resulting_array)
                    sys.stderr.write('\rdone {0:%}'.format(i/len(input_arrays)))
        else:
            for i, input_array in enumerate(input_arrays):
                array_list.append(lcs(input_array, verbose=True))
                sys.stderr.write('\rdone {0:%}'.format(i / len(input_arrays)))

        eigenvalues = xr.concat(array_list, dim='time')

        # from xrviz.dashboard import Dashboard
        # dashboard = Dashboard(eigenvalues)
        #dashboard.show()
        # u.time
        #eigenvalues = xr.apply_ufunc(lambda x, y: lcs(u=x, v=y), u.groupby('time'), v.groupby('time'), dask='parallelized')
        #eigenvalues = ds.groupby('time').apply(lcs)
        return eigenvalues




if __name__ == '__main__':

    classifier = Classifier()

    running_on = str(sys.argv[1])
    lcs_type = str(sys.argv[2])
    year = str(sys.argv[3])
    lcs_time_len = 1  # * 6 hours intervals
    #running_on = ''
    #lcs_type = 'repelling'
    #year = 2000
    config['array_slice']['time'] = slice(f'{year}-01-01T00:00:00', f'{year}-12-31T18:00:00')
    config['u_filename'] = f'viwve_ERA5_6hr_{year}010100-{year}123118.nc'
    config['v_filename'] = f'viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
    config['tcwv_filename'] = f'tcwv_ERA5_6hr_{year}010100-{year}123118.nc'


    #running_on =''
    #lcs_type = 'attracting'
    if running_on == 'jasmin':
        config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
        outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
        #outpath = '/home/users/gmpp/'
    else:
        outpath = 'data/'

    classified_array1 = classifier(config, method='lagrangian', lcs_type=lcs_type, lcs_time_len=lcs_time_len)
    print("*---- Saving file ----*")
    classified_array1.to_netcdf(f'{outpath}SL_{lcs_type}_{year}_lcstimelen_{lcs_time_len}.nc')
