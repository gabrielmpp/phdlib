import xarray as xr
from meteomath import to_cartesian, divergence
import pandas as pd
from convlib.LCS import LCS, parcel_propagation
import sys
from typing import Optional
import numpy as np
import concurrent.futures
from convlib.xr_tools import get_seq_mask, get_xr_seq


config = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_2000010100-2000123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_2000010100-2000123118.nc',
    'tcwv_filename': 'tcwv_ERA5_6hr_2000010100-2000123118.nc',
    'time_freq': '6H',
    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-02-07T18:00:00'),
                   'latitude': slice(15, -50),
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

        pass

    def __call__(self, config: dict, method: str, lcs_type: Optional[str] = None, lcs_time_len: Optional[int] = 4,
                 find_departure: bool = False, parallel: bool = False) -> xr.DataArray:
        """

        :rtype: xr.DataArray
        """
        print(f"*---- Calling classifier with method {method} ----*")
        self.method = method
        self.config = config
        print("Apply classifier in the following domain")
        print(config['array_slice'])
        self.parallel = parallel

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
            classified_array = self._lagrangian_method(u, v, lcs_type, lcs_time_len, find_departure=find_departure)
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
        u = xr.open_dataarray(self.config['data_basepath'] + self.config['u_filename'])
        v = xr.open_dataarray(self.config['data_basepath'] + self.config['v_filename'])
        tcwv = xr.open_dataarray(self.config['data_basepath'] + self.config['tcwv_filename'])

        u.coords['longitude'].values = (u.coords['longitude'].values + 180) % 360 - 180
        v.coords['longitude'].values = (v.coords['longitude'].values + 180) % 360 - 180
        tcwv.coords['longitude'].values = (tcwv.coords['longitude'].values + 180) % 360 - 180

        u = u.sel(self.config['array_slice'])
        v = v.sel(self.config['array_slice'])
        tcwv = tcwv.sel(self.config['array_slice'])


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

    def _lagrangian_method(self, u, v, lcs_type: str, lcs_time_len=4, find_departure=False) -> xr.DataArray:
        parallel = self.parallel
        time_dir = 'backward'
        u = get_xr_seq(u, 'time', [x for x in range(lcs_time_len)])
        u = u.dropna(dim='time',how='any')
        v = get_xr_seq(v, 'time', [x for x in range(lcs_time_len)])
        v = v.dropna(dim='time',how='any')
        print(u)
        #####
        ####
        #####
        # get seq mask does not seem to be working nicely
        #v = get_seq_mask(v, 'time', lcs_time_len)
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
        ds_groups = list(ds.groupby('time'))
        input_arrays = []
        for label, group in ds_groups: # have to do that because bloody groupby returns the labels
            input_arrays.append(group)

        lcs = LCS(lcs_type=lcs_type, timestep=timestep, timedim='seq', shearless=shearless)
        print(input_arrays[0])
        if find_departure:
            x_list = []
            y_list = []
            if parallel:
                raise Exception("Parallel not implemented")
                pass
            #    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            #       for i, resulting_array in enumerate(executor.map(lcs, input_arrays)):
            #           u_list.append(resulting_array[0])
            #           v_list.append(resulting_array[1])
            #          sys.stderr.write('\rdone {0:%}'.format(i/len(input_arrays)))
            else:
                for i, input_array in enumerate(input_arrays):
                    x_departure, y_departure = parcel_propagation(input_array.u, input_array.v, timestep)
                    x_list.append(x_departure)
                    y_list.append(y_departure)
                    sys.stderr.write('\rdone {0:%}'.format(i / len(input_arrays)))
                x_list = xr.concat(x_list, dim='time')
                y_list = xr.concat(y_list, dim='time')
                x_list.name = 'x_departure'
                y_list.name = 'y_departure'
                output = xr.merge([x_list, y_list])
        else:
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

            output = xr.concat(array_list, dim='seq')

        return output



if __name__ == '__main__':

    classifier = Classifier()
    parallel = False
    find_departure = False
    running_on = str(sys.argv[1])
    lcs_type = str(sys.argv[2])
    year = str(sys.argv[3])
    lcs_time_len = 4  # * 6 hours intervals
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

    classified_array1 = classifier(config, method='lagrangian', lcs_type=lcs_type, lcs_time_len=lcs_time_len,
                                   find_departure=find_departure, parallel=parallel)
    print("*---- Saving file ----*")
    if find_departure:
        classified_array1.to_netcdf(f'{outpath}SL_{lcs_type}_{year}_departuretimelen_{lcs_time_len}.nc')
    else:
        classified_array1.to_netcdf(f'{outpath}SL_{lcs_type}_{year}_lcstimelen_{lcs_time_len}.nc')
