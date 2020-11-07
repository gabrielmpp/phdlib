import xarray as xr
import pandas as pd
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.trajectory import parcel_propagation
import glob
from typing import Optional
from xr_tools.tools import latlonsel, get_xr_seq_coords_only
import numpy as np
import uuid
import datetime
import os
import subprocess

config_jasmin = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_{year}010100-{year}123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc',
    'tcwv_filename': 'tcwv_ERA5_6hr_{year}010100-{year}123118.nc',
    'time_freq': '6H',
    'chunks': {
        'time': 100, }
        ,
    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-02-07T18:00:00'),
                   'latitude': slice(-40, -20),
                   'longitude': slice(-50, -30),
                   # 'latitude': slice(-20, -35),
                   # 'longitude': slice(-55, -35)
                    },

    'array_slice_latlon': {
        'latitude': slice(-75, 45),
        'longitude': slice(-170, 30),
        # 'latitude': slice(-20, -35),
        # 'longitude': slice(-55, -35)
    },
    'array_slice_time': {
        'time': slice(None, None),
    }
    }

config_local = {
    'data_basepath': '/home/gab/phd/data/ERA5/',
    'u_filename': 'ERA5viwve_ERA5_6hr_{year}010100-{year}123118.nc',
    'v_filename': 'ERA5viwvn_ERA5_6hr_{year}010100-{year}123118.nc',
    'tcwv_filename': 'ERA5tcwv_ERA5_6hr_{year}010100-{year}123118.nc',
    'time_freq': '6H',
    'chunks': {
        'time': 40}
        ,
    'array_slice_latlon': {
                   'latitude': slice(-50, -15),
                   'longitude': slice(-65, -30),
                   # 'latitude': slice(-20, -35),
                   # 'longitude': slice(-55, -35)
                    },
    'array_slice_time': {
    'time': slice(None, 40),
    }
    }


class Classifier:
    """
    Convergence zones classifier
    """

    def __init__(self, config: dict, method: str, lcs_time_len: Optional[int] = 4,
                 find_departure: bool = False, parallel: bool = False, init_time: Optional[int] = None,
                 final_time: Optional[int] = None):
        """

        :param config: Dictionary with
        :param method: Classification method
        :param lcs_type: LCS type
        :param lcs_time_len:
        :param find_departure:
        :param parallel:
        :param init_time: Only necessary for limited time tests - Set to None for complete runs
        :param final_time: same as init_time
        :return:
        """
        self.init_time = init_time
        self.final_time = final_time
        self.method = method
        self.config = config
        self.parallel = parallel
        self.lcs_time_len = lcs_time_len
        self.find_departure = find_departure

    def __call__(self) -> xr.DataArray:

        print(f"*---- Calling classifier with method {self.method} ----*")

        u, v = self._read_data
        # u = to_cartesian(u)
        # v = to_cartesian(v)
        print("*---- Applying classification method ----*")

        if self.method == 'Q':
            classified_array = self._Q_method(u, v)
        elif self.method == 'conv':
            classified_array = self._conv_method(u, v)
        elif self.method == 'lagrangian':
            assert isinstance(lcs_type, str), 'lcs_type must be string'
            classified_array = self.call_lcs(u, v, lcs_type, lcs_time_len, find_departure=find_departure)
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
        timeslice = self.config['array_slice_time']['time']
        years = pd.date_range(timeslice.start, timeslice.stop, freq='Y')
        u_filenames = []
        v_filenames = []
        tcwv_filenames = []
        for year in years.year.values:
            u_filenames.append(self.config['data_basepath'] + self.config['u_filename'].format(year=year))
            v_filenames.append(self.config['data_basepath'] + self.config['v_filename'].format(year=year))
            tcwv_filenames.append(self.config['data_basepath'] + self.config['tcwv_filename'].format(year=year))

        u = xr.open_mfdataset(u_filenames, chunks=self.config['chunks'])
        v = xr.open_mfdataset(v_filenames, chunks=self.config['chunks'])
        tcwv = xr.open_mfdataset(tcwv_filenames, chunks=self.config['chunks'])
        u = u.to_array().isel(variable=0).drop('variable')
        v = v.to_array().isel(variable=0).drop('variable')
        tcwv = tcwv.to_array().isel(variable=0).drop('variable')

        u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
        v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
        tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
        u = latlonsel(u, **self.config['array_slice_latlon'])
        v = latlonsel(v, **self.config['array_slice_latlon'])
        tcwv = latlonsel(tcwv, **self.config['array_slice_latlon'])
        assert pd.infer_freq(u.time.values) == pd.infer_freq(v.time.values), "u and v should have equal time frequencies"
        data_time_freq = pd.infer_freq(u.time.values)

        if data_time_freq != self.config['time_freq']:
            print("Resampling data to {}".format(self.config['time_freq']))
            u = u.resample(time=self.config['time_freq']).interpolate('linear')
            v = v.resample(time=self.config['time_freq']).interpolate('linear')
            tcwv = tcwv.resample(time=self.config['time_freq']).interpolate('linear')

        if 'viwv' in self.config['u_filename']:
            print("Applying unit conversion")
            u = u / tcwv
            v = v / tcwv
            print("Done unit conversion")

        # new_lon = np.linspace(u.longitude[0].values, u.longitude[-1].values, int(u.longitude.values.shape[0] * 0.5))
        # new_lat = np.linspace(u.latitude[0].values, u.latitude[-1].values, int(u.longitude.values.shape[0] * 0.5))
        # print("*---- NOT Start interp ----*")
        # u = u.interp(latitude=new_lat, longitude=new_lon)
        # v = v.interp(latitude=new_lat, longitude=new_lon)
        # print('*---- Finish interp ----*')
        print("*---- Done reading ----*")
        print(u)

        return u, v

    def preprocess_and_save(self, lcs_time_len) -> None:

        u, v = self._read_data
        u.name = 'u'
        v.name = 'v'

        ds = xr.merge([u, v])
        ds = ds.sel(config['array_slice_time'])
        timess = get_xr_seq_coords_only(ds, 'time', idx_seq=np.arange(lcs_time_len))

        script_path = '/home/users/gmpp/phdscripts/phdlib/convlib/slurm_submission.sh'

        timestep = -6 * 3600
        for time in timess.time.values:
            print('Writing time {}'.format(time))
            times = timess.sel(time=time).values
            input = ds.sel(time=times)
            savepath = f'{outpath_temp}input_partial_{time}.nc'
            ftlepath = f"{outpath_temp}SL_attracting_lcstimelen_{config['lcs_time_len']}_partial_{time}.nc"
            input.to_netcdf(savepath)

            # Args: timestep, timedim, SETTLS_order, subdomain, ds_path, outpath
            subprocess.call(['sbatch',
                              script_path, str(timestep), 'time', str(4),
                             '-85/-32/-40/15', savepath, ftlepath]
                            )


if __name__ == '__main__':

    parallel = True
    find_departure = False
    # running_on = str(sys.argv[1])
    # lcs_type = str(sys.argv[2])
    # start_year = str(sys.argv[3])
    # end_year = str(sys.argv[3])
    # lcs_time_len = int(sys.argv[4]) # * 6 hours intervals
    # running_on = 'jasmin'
    continue_old_run = False
    if not continue_old_run:
        lcs_type = 'attracting'

        lcs_time_len = 1
        start_year = 1981
        end_year = 2009
        config = config_jasmin
        config['start_year'] = start_year
        config['end_year'] = end_year
        config['lcs_time_len'] = lcs_time_len

        config['array_slice_time']['time'] = slice(f'{start_year}-01-01T00:00:00', f'{end_year}-12-31T18:00:00')
        config['start_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        running_on = 'jasmin'
        if running_on == 'jasmin':
            config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
            # outpath_temp = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_{}/'.format(uuid.uuid4())
            outpath_temp = '/work/scratch-nopw/gmpp/experiment_timelen_{timelen}_{id}/'.format(id=uuid.uuid4(),
                                                                                         timelen=str(lcs_time_len), end_year=str(end_year))

        else:
            config['data_basepath'] = '/home/gab/phd/data/ERA5/'
            outpath_temp = '/home/gab/phd/data/FTLE_ERA5/experiment_{}/'.format(uuid.uuid4())
        print(outpath_temp)
        os.mkdir(outpath_temp)
        with open(outpath_temp + 'config.txt', 'w') as f:
            f.write(str(config))
    else:
        outpath_temp = '/work/scratch-pw/gmpp/experiment_timelen_4_ea4f4422-7f91-4f94-9d33-dfe9d23987c7/'
        with open(outpath_temp + 'config.txt') as f:
            conf = f.read()
            config = eval(conf)
        list_of_files = glob.glob(outpath_temp + 'SL*')  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        latest_date = latest_file.split('partial_')[1].split('.nc')[0]
        #  Now need to account for offset due to propagation interval
        dateee = pd.Timestamp(latest_date) - pd.Timedelta(str(config['lcs_time_len'] * 6) + 'H')  # 6 is the time interval
        dateee = str(np.datetime64(dateee))
        config['array_slice_time']['time'] = slice(dateee, f"{config['end_year']}-12-31T18:00:00")
        print(config['array_slice_time']['time'])

    classifier = Classifier(config, method='lagrangian', lcs_time_len=config['lcs_time_len'],
               find_departure=find_departure, parallel=parallel)

    classifier.preprocess_and_save(config['lcs_time_len'])