import xarray as xr
from meteomath import  divergence
import pandas as pd
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.trajectory import parcel_propagation
import sys
from typing import Optional
import concurrent.futures
from xr_tools.tools import  get_xr_seq, latlonsel

config_jasmin = {
    'data_basepath': '/media/gabriel/gab_hd/data/sample_data/',
    'u_filename': 'viwve_ERA5_6hr_{year}010100-{year}123118.nc',
    'v_filename': 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc',
    'tcwv_filename': 'tcwv_ERA5_6hr_{year}010100-{year}123118.nc',
    'time_freq': '6H',
    'chunks': {
        'time': 1000, }
        ,
    'array_slice': {'time': slice('2000-02-06T00:00:00', '2000-02-07T18:00:00'),
                   'latitude': slice(-80, 30),
                   'longitude': slice(-180, 15),
                   # 'latitude': slice(-20, -35),
                   # 'longitude': slice(-55, -35)
                    },

    'array_slice_latlon': {
        'latitude': slice(-80, 30),
        'longitude': slice(-180, -1),
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

    def __init__(self):

        pass

    def __call__(self, config: dict, method: str, lcs_type: Optional[str] = None, lcs_time_len: Optional[int] = 4,
                 find_departure: bool = False, parallel: bool = False, init_time: Optional[int] = None,
                 final_time: Optional[int] = None) -> xr.DataArray:
        """

        :param config: Dictionary with
        :param method: Classification method
        :param lcs_type: LCS type
        :param lcs_time_len:
        :param find_departure:
        :param parallel:
        :param init_time: Only necessary for limited time tests - Set to None for complete runs
        :param final_time: same as init_time
        :param subtimes_len:
        :return:
        """
        print(f"*---- Calling classifier with method {method} ----*")
        self.init_time = init_time
        self.final_time = final_time
        self.method = method
        self.config = config
        self.parallel = parallel

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
        print("*---- NOT Start interp ----*")
        # u = u.interp(latitude=new_lat, longitude=new_lon)
        # v = v.interp(latitude=new_lat, longitude=new_lon)
        print('*---- Finish interp ----*')


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
        print(f"lcs_time_len {lcs_time_len}")
        print( [x for x in range(lcs_time_len)])
        parallel = self.parallel
        SETTLS_order = 2
        time_dir = 'backward'
        print('get_xr_seq')
        u = get_xr_seq(u, 'time', [x for x in range(lcs_time_len)])
        u = u.dropna(dim='time', how='any')
        v = get_xr_seq(v, 'time', [x for x in range(lcs_time_len)])
        v = v.dropna(dim='time', how='any')

        shearless = False

        timestep = self.config['time_freq']
        if 'H' in timestep:
            timestep = float(timestep.replace('H', '')) * 3600
        elif 'D' in timestep:
            timestep = float(timestep.replace('D', '')) * 86400
        else:
            raise ValueError(f"Frequency {timestep} not supported.")

        timestep = -timestep if time_dir == 'backward' else timestep
        # self.config['chunks']['seq'] = lcs_time_len

        # u = u.chunk(self.config['chunks'])
        # v = v.chunk(self.config['chunks'])
        # # newchunks = dict(
        # #     time=None,
        # #     latitude=4,
        # #     longitude=4
        # # )
        # u = u.chunk(newchunks)
        # v = v.chunk(newchunks)

        u.name = 'u'
        v.name = 'v'
        lcs = LCS(lcs_type=lcs_type, timestep=timestep, timedim='seq', shearless=shearless, SETTLS_order=SETTLS_order)

        ds = xr.merge([u, v])
        ds = ds.chunk(dict(seq=None))
        # a = ds.groupby('time').apply(lcs)
        # def run_lcs(ds, lcs):
        #     return xr.apply_ufunc(lambda x, y: lcs(u=x, v=y), ds.u, ds.v, input_core_dims=[['seq'], ['seq']],
        #                           dask='parallelized', output_dtypes=[float])
        # a = run_lcs(ds, lcs).compute()
        ntimes = ds.time.shape[0]
        import os


        # processes = os.cpu_count() - 1
        processes = 5
        nbins = int(ntimes/processes)
        print('grouping')
        ds_groupss = list(ds.groupby_bins('time', bins=nbins))
        input_arrayss = []
        for label, group in ds_groupss: # have to do that because bloody groupby returns the labels
            input_arrayss.append(group)

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
                    x_departure, y_departure = parcel_propagation(input_array.u.copy(), input_array.v.copy(), timestep,
                                                                  propdim='seq',
                                                                  SETTLS_order=SETTLS_order)
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
            for input_arrays in input_arrayss:
                ds_groups = input_arrays.groupby('time')
                ds_list = []
                for label, group in ds_groups:
                    ds_list.append(group.load())
                if parallel:
                    print('Starting parallel jobs')
                    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
                        for i, resulting_array in enumerate(executor.map(lcs, ds_list)):
                            array_list.append(resulting_array.chunk(dict(seq=1)))
                            sys.stderr.write('\rdone {0:%}'.format(i/len(input_arrays)))
                else:
                    for i, input_array in enumerate(ds_list):
                       array_list.append(lcs(input_array, verbose=True))
                       sys.stderr.write('\rdone {0:%}'.format(i / len(input_arrays)))


            output = xr.concat(array_list, dim='seq')

            # idx = - lcs_time_len + 1 if lcs_time_len > 1 else None

            output.seq.values = u.time.values
            output = output.rename({'seq': 'time'})

        return output



if __name__ == '__main__':

    classifier = Classifier()
    parallel = True
    find_departure = False
    # running_on = str(sys.argv[1])
    # lcs_type = str(sys.argv[2])
    # start_year = str(sys.argv[3])
    # end_year = str(sys.argv[3])
    # lcs_time_len = int(sys.argv[4]) # * 6 hours intervals
    running_on = 'jasmin'
    lcs_type = 'attracting'
    start_year = 1980
    end_year = 2009
    lcs_time_len = int(sys.argv[1])
    config = config_jasmin

    config['array_slice_time']['time'] = slice(f'{start_year}-01-01T00:00:00', f'{end_year}-12-31T18:00:00')

    #running_on =''
    #lcs_type = 'attracting'
    if running_on == 'jasmin':
        config['data_basepath'] = '/gws/nopw/j04/primavera1/observations/ERA5/'
        outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
        # outpath = '/home/users/gmpp/'
    else:
        outpath = '/home/gab/phd/data/FTLE_ERA5/'

    classified_array1 = classifier(config, method='lagrangian', lcs_type=lcs_type, lcs_time_len=lcs_time_len,
                                   find_departure=find_departure, parallel=parallel)
    print("*---- Saving file ----*")
    if find_departure:
        classified_array1.to_netcdf(f'{outpath}SL_{lcs_type}_{start_year}_{end_year}_departuretimelen_{lcs_time_len}_v2.nc')
    else:
        classified_array1.to_netcdf(f'{outpath}SL_{lcs_type}_{start_year}_{end_year}_lcstimelen_{lcs_time_len}_v2.nc')
