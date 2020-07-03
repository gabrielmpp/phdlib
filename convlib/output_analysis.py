import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import dask
from multiprocessing.pool import ThreadPool
# dask.config.set(scheduler='threads') # Global config
# dask.config.set(pool=ThreadPool(80)) # To avoid omp error
import gc
import threading
import glob
from xr_tools.tools import common_index

import xarray as xr
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import bottleneck
# from scipy import stats
from xr_tools.tools import safely_read_multiple_files
from xr_tools.tools import latlonsel
from scipy.signal import savgol_filter
import cmasher as cmr


def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, keepdims=True))
            * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)


def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


def spearman_correlation_gufunc(x, y):
    x_ranks = bottleneck.rankdata(x, axis=-1)
    y_ranks = bottleneck.rankdata(y, axis=-1)
    return pearson_correlation_gufunc(x_ranks, y_ranks)


def spearman_correlation(x, y, dim):
    return xr.apply_ufunc(
        spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='allowed',
        output_dtypes=[float])


def spearman_pvalue(x, y, dim):
    return xr.apply_ufunc(lambda a, b: stats.spearmanr(a, b)[1], x, y,
                          input_core_dims=[[dim], [dim]],
                          dask='allowed',
                          output_dtypes=[float])


def read_u_v(years):
    u_list, v_list, z_list = [], [], []
    for year in np.arange(years.start, years.stop):

        u = xr.open_dataset(datapath + ufilename.format(year=year), chunks={'time': 100})
        v = xr.open_dataset(datapath + vfilename.format(year=year), chunks={'time': 100})
        u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
        v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
        u = u.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        v = v.sel(latitude=slice(20, -60), longitude=slice(-80, -10))

        u = u.to_array().isel(variable=0).drop('variable')
        v = v.to_array().isel(variable=0).drop('variable')
        u_list.append(u)
        v_list.append(v)
        print(str(year))
    u = xr.concat(u_list, dim='time')
    v = xr.concat(v_list, dim='time')
    [x.close() for x in u_list]
    [x.close() for x in v_list]
    u_list = None
    v_list = None
    return u, v

# outpath = '/home/gab/phd/data/FTLE_ERA5/'
outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
experiments = [
    # 'experiment_timelen_12_d8513a9b-3fd6-4df4-9b05-377a9d8e64ca/',
    # 'experiment_timelen_16_105eee10-9804-4167-b151-9821c41136f6/',
    # 'experiment_timelen_4_5cc17190-9174-4642-b70c-a6170a808eb5/',
    'experiment_timelen_8_c102ec42-2a6f-4c98-be7d-31689c6c60a9/'
]

regions = {
    'sp': {
        'latitude': slice(-25, -20),
        'longitude': slice(-47, -43)
    },
    'llj': {
        'latitude': slice(-15, -10),
        'longitude': slice(-65, -60)
    }
    ,
    'n_litoral': {
        'latitude': slice(-3, 3),
        'longitude': slice(-60, -55)
    }
}


def calc_duration(ts, days):
    peaks = find_peaks(ts, height=1.2, distance=days * 4)[0]
    widths = peak_widths(ts, peaks, rel_height=0.5)
    return ts.isel(time=0).drop('time').copy(data=np.mean(widths[0] / days))


years = slice(1981, 1989)
datapath = '/gws/nopw/j04/primavera1/observations/ERA5/'
vfilename = 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
ufilename = 'viwve_ERA5_6hr_{year}010100-{year}123118.nc'

# events = da.sel(latitude=-23.5, longitude=-46.5)
# events = events.where(events > 1, 0)
# events = events.where(events < 1, 1)
# da_events = da.where(events==1, drop=True)
# da_events = da_events.mean('time')
# da_ = da.groupby('time.month').mean('time', skipna=True)
# da_ = da.groupby('time.month').var('time')
# month=1
# wsheds = da_.sel(month=month).copy(data=watershed(-da_.sel(month=month).values, markers=30,))
# da_.sel(month=month).plot()
# wsheds.plot.contour()


# --- Reading convzones data --- #
# for experiment in experiments:
#     print('!')
#     print(('Number of threads: ' + str(threading.active_count())))
#     with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     files = [f for f in glob.glob(outpath + experiment + "**/*.nc", recursive=True)]
#     da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})
#     da = da.isel(time=slice(0, 6000))
#     # da = da.resample(time='1D').mean()
#     da = da.compute()
#     da = np.log(np.sqrt(da)) / days
#     events = da.where(da > 1, 0)
#     events = events.where(events < 1, 1)
#     from scipy.ndimage import label, find_objects,generate_binary_structure
#     # from skimage.measure import label
#     from skimage.measure import regionprops_table
#
#     s = generate_binary_structure(3,  connectivity=1)
#     events_labeled = events.copy(data=label(events.sel(latitude=slice(-30, None)).values, structure=s)[0])
#     props = pd.DataFrame(regionprops_table(
#         events_labeled.values, intensity_image=da.values,
#         properties=['slice', 'mean_intensity', 'major_axis_length', 'area', 'centroid'])
#     )
#     props = props.iloc[1:, :]
#     props['centroid-1'] = da.latitude.isel(
#         latitude=props['centroid-1'].round(0).values.astype(int)
#     )
#     props['centroid-2'] = da.longitude.isel(
#         longitude=props['centroid-2'].round(0).values.astype(int)
#     )
#     props['time_index'] = [int(0.5*(p[0].start + p[0].stop))for p in props['slice']]
#     props['slice'] = [p[0].stop - p[0].start for p in props['slice']]
#     props = props.rename(columns={'slice': 'time duration',
#                                   'centroid-1': 'latitude',
#                                   'centroid-2': 'longitude'})
#     props = props[props['longitude'] > -60] #removing andes
#     res_deg = da.latitude.diff('latitude')[0].values
#     props['major_axis_length'] = props['major_axis_length'] * res_deg
#     props['istropical'] = [True if 5> x > -5 else False for x in props['latitude']]
#     props['issubtropical'] = [True if -5 > x > -20 else False for x in props['latitude']]
#     def latitudinal_band(x):
#         if 5 > x > -5 :
#             return 'Tropical'
#         elif -5 > x > -20:
#             return 'Subtropical'
#         elif x < -20:
#             return 'Extratropical'
#
#
#     props['Latitude band'] = [latitudinal_band(x) for x in props['latitude']]
#
#     latitude_band = 'Extratropical'
#     subtropical_props = props[props['Latitude band'] == latitude_band]
#     # raise Exception('stop')
#
#     time_idx = np.unique(subtropical_props[subtropical_props['time duration'] == 2]['time_index'].values)
#     anomaly = da.isel(time=time_idx).mean('time') -\
#               da.mean('time')
#     fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
#     anomaly.plot(ax=axs[0], transform=ccrs.PlateCarree(), vmin=-0.1, vmax=0.1, cmap='RdBu',
#                  add_colorbar=False)
#     axs[0].coastlines()
#     axs[0].axhline(-20)
#     axs[0].axhline(-5)
#     axs[0].axhline(5)
#     plt.close()
#
#     fig, axs = plt.subplots(3, 3, figsize=[10, 10], sharey='row', sharex='col',
#                             gridspec_kw={'wspace': 0.07, 'hspace': 0.07})
#     ec = None
#     bintype='scott'
#     props[props['Latitude band'] == 'Tropical'][['time duration']].hist(bins = bintype, width=0.8, ec=ec,ax=axs[0, 0], alpha=0.7, zorder=10)
#     props[props['Latitude band'] == 'Subtropical'][['time duration']].hist(bins = bintype, width=0.8, ec=ec,ax=axs[1, 0], alpha=0.7, zorder=5)
#     props[props['Latitude band'] == 'Extratropical'][['time duration']].hist(bins = bintype, width=0.8, ec=ec,ax=axs[2, 0], alpha=0.7, zorder=5)
#
#     props[props['Latitude band'] == 'Tropical'][['mean_intensity']].hist(bins = bintype, ec=ec,ax=axs[0, 1], alpha=0.7, zorder=5)
#     props[props['Latitude band'] == 'Subtropical'][['mean_intensity']].hist(bins = bintype, ec=ec,ax=axs[1, 1], alpha=0.7, zorder=10)
#     props[props['Latitude band'] == 'Extratropical'][['mean_intensity']].hist(bins = bintype, ec=ec,ax=axs[2, 1], alpha=0.7, zorder=10)
#
#     props[props['Latitude band'] == 'Tropical'][['major_axis_length']].hist(bins = bintype, ec=ec,ax=axs[0, 2], alpha=0.7, zorder=10)
#     props[props['Latitude band'] == 'Subtropical'][['major_axis_length']].hist(bins = bintype, ec=ec,ax=axs[1, 2], alpha=0.7, zorder=5)
#     props[props['Latitude band'] == 'Extratropical'][['major_axis_length']].hist(bins = bintype,ec=ec,ax=axs[2, 2], alpha=0.7, zorder=5)
#     for ax in axs.flatten(): ax.semilogy(True), ax.set_ylim([0, 1e4])
#     for ax in axs[:, 0]: ax.set_xlim([0, 15])
#     for ax in axs[:, 1]: ax.set_xlim([1, 2])
#     for ax in axs[:, 2]: ax.set_xlim([0, 30])
#     for ax in axs.flatten()[3:]: ax.set_title(None)
#     titles = ['Tropical', 'Subtropical', 'Extratropical']
#     for i, ax in enumerate(axs[:, 0]): ax.set_ylabel(titles[i])
#     axs[0, 1].set_xlim([1, None])
#     axs[0, 0].set_xlim([1, None])
#     axs[0, 2].set_xlim([0, None])
#     axs[1, 1].set_xlim([1, None])
#     axs[1, 0].set_xlim([1, None])
#     axs[1, 2].set_xlim([0, None])
#     axs[2, 1].set_xlim([1, None])
#     axs[2, 0].set_xlim([1, None])
#     axs[2, 2].set_xlim([0, None])
#
#
#     plt.savefig('tempfigs/panel_histograms.pdf')
#     plt.close()
#
#
#     time_idx = np.unique(subtropical_props[subtropical_props['time duration'] == 3]['time_index'].values)
#     anomaly = da.isel(time=time_idx).mean('time') -\
#               da.mean('time')
#     anomaly.plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=-0.1, vmax=0.1, cmap='RdBu',
#                  add_colorbar=False)
#     axs[1].coastlines()
#     axs[1].axhline(-20)
#     axs[1].axhline(-5)
#     axs[1].axhline(5)
#     plt.savefig(f'tempfigs/anomalies_{latitude_band}.pdf')
#     plt.close()
#     subtropical_props = props[props['Latitude band'] == latitude_band]
#     p = pd.plotting.scatter_matrix(subtropical_props[['time duration', 'mean_intensity', 'major_axis_length']], diagonal='kde',
#                                c=subtropical_props['longitude'])
#     plt.savefig(f'tempfigs/scatter_{latitude_band}.pdf')
#     plt.close()
#
#     p = subtropical_props['time duration'].hist()
#     p.semilogy(True)
#     p.set_xlabel('Duration (days)')
#     p.set_ylabel('Count')
#     plt.savefig(f'tempfigs/hist_duration_{latitude_band}.pdf')
#     plt.close()
#
#     props.plot.hexbin(y='latitude', x='longitude', C='time duration', vmax=4, cmap='jet')
#     plt.style.use('bmh')
#     fig, axs = plt.subplots(1, 2, figsize=[20, 7])
#     props[props['istropical']].plot.hexbin(x='time duration', y='major_axis_length', cmap='jet', C='mean_intensity',
#                                            gridsize=[20, 10], vmax=1.2,  ax=axs[0], xlim=[0, 10], ylim=[0, 30])
#     props[props['issubtropical']].plot.hexbin(x='time duration', y='major_axis_length', cmap='jet', C='mean_intensity',
#                                             gridsize=[5, 5], vmax=1.2,   ax=axs[1], xlim=[0, 10], ylim=[0, 30])
#     axs[0].scatter(x=props[props['istropical']]['time duration'], y=props[props['istropical']]['major_axis_length'], alpha=0.5)
#     axs[1].scatter(x=props[props['issubtropical']]['time duration'], y=props[props['issubtropical']]['major_axis_length'], alpha=0.5)
#
#
#     plt.savefig('tempfigs/hexplot.png')
#
#     p = props[['time duration']].hist(by=props['Latitude band'], alpha=0.5, bins=10)
#     for ax in p.flatten(): ax.semilogy(True)
#     tropical_props = props[props['Latitude band'] == 'Tropical']
#
#
#     p = pd.plotting.scatter_matrix(tropical_props[['time duration', 'mean_intensity', 'major_axis_length']],
#                                c=tropical_props['longitude'])
#     p[1, 0].set_xlim([0, 10])
#     p[2, 0].set_xlim([0, 10])
#     p[0, 1].set_ylim([0, 10])
#     p[0, 2].set_ylim([0, 10])
#     p[0, 0].semilogy(True)
#     p[1, 1].semilogy(True)
#     p[2, 2].semilogy(True)
#     plt.savefig('scatter_tropical.pdf')
#
#
#
#
#


# props.plot.scatter(x='time duration', y='major_axis_length')
#
# p = plt.scatter(x=props['time duration'], y=props['major_axis_length'],
#                 c=props['latitude'], s=30 * props['mean_intensity'],
#                 alpha=0.5)
# plt.xlabel('Event duration (days)')
# plt.ylabel('Major axis length (degrees)')
# plt.legend(['< -5', '>-5'], title='Latitude slice')
# x_coords = [int(prop.centroid[2]) for prop in props]
# for prop in props:
#     plt.scatter(prop.centroid[2], prop.centroid[1])
#
# events_labeled = events.copy(data=label(events.values)[0])
# objs = find_objects(events_labeled.values)
# obj_durations = []
# for i, obj in enumerate(objs):
#     print(i/len(objs))
#     time_size = obj[0].stop - obj[0].start
#     obj_durations.append(time_size)
# obj_durations = np.array(obj_durations)
# objs = np.array(objs)
# long_objs = objs[np.where(obj_durations > 4)]
# l = []
# for long_obj in long_objs[1:]:
#     l.append(events.isel({events.dims[0]: long_obj[0], events.dims[1]: long_obj[1], events.dims[2]: long_obj[2]}))
#
# l_mean = l[0].mean('time')
# for l_ in l: l_.mean('time').plot.contour()
#
#
# events.isel({events.dims[0]: obj[0], events.dims[1]: obj[1], events.dims[2]: obj[2]})
# pt_sp = da.sel(latitude=-23.5, longitude=-46.5)
# da = da.compute()
# da = da.chunk({'latitude': 10, 'longitude': 10})
# da = da.stack({'points': ['latitude', 'longitude']})
# dur = da.groupby('points').apply(calc_peaks_and_duration, days=days)

# find_peaks(pt_sp, height=1.2, distance=days)[0]
#
# da_groups = list(da.groupby('time.year'))
#
# months = [pd.Timestamp(x).strftime('%m') for x in da.time.values]
# djf_mask = np.array([True if x == '01' or x == '02' or x == '12' else False for x in months])
# jja_mask = np.array([True if x == '06' or x == '07' or x == '08' else False for x in months])
# son_mask = np.array([True if x == '09' or x == '10' or x == '11' else False for x in months])
# mam_mask = np.array([True if x == '03' or x == '04' or x == '05' else False for x in months])
# da_djf = da.sel(time=da.time.values[djf_mask])
# da_son = da.sel(time=da.time.values[son_mask])
# da_mam = da.sel(time=da.time.values[mam_mask])
# da_jja = da.sel(time=da.time.values[jja_mask])
# da_seasons = xr.concat([da_djf, da_jja, da_son, da_mam], dim=pd.Index(['DJF', 'JJA', 'SON', 'MAM'], name='season'))
# years = [pd.Timestamp(x).strftime('%Y') for x in da.time.values]
# for year in years:
#     da_year = da_seasons.sel(time=year)
#     duration_year =
# dur = da.groupby('points').apply(calc_duration, days=days)
# dur = dur.unstack()
# da = da.unstack()
# fig, ax = plt.subplots( subplot_kw={'projection': ccrs.PlateCarree()})
# dur.where(events.mean('time') < 0.4).where(events.sum('time') > 10).plot(
#     ylim=[-30, 5], xlim=[-65, -34], ax=ax, vmax=4, cmap='nipy_spectral')
# ax.coastlines()
# plt.show()
# pt_sp = pt_sp.sel(time=slice('1987-01', '1987-06'))
#
# plt.style.use('bmh')
# peaks = find_peaks(pt_sp, height=1, distance=days*4)[0]
# widths = peak_widths(pt_sp, peaks=peaks, rel_height=0.5)
# np.mean(widths[0]/days)
# widths[0]
# widths[0]
# plt.plot(pt_sp.values)
# plt.plot(peaks, pt_sp.values[peaks], 'x', color='red')
# plt.hlines(*widths[1:])
# plt.xticks(np.arange(pt_sp.time.values.shape[0], step=30),
#            [pd.Timestamp(x).strftime('%Y-%m') for x in pt_sp.time.values] )
# plt.ylabel('FTLE')
# plt.xlabel('Date')


#
u, v = read_u_v(years)

MAG = xr.open_dataset('~/phdscripts/phdlib/convlib/data/xarray_mair_grid_basins.nc')
MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})
dlat = np.sign(MAG['amazon'].diff('latitude'))
dlon = np.sign(MAG['amazon'].diff('longitude'))
border_amazon = np.abs(dlat) + np.abs(dlon)
border_amazon = border_amazon.where(border_amazon <= 1, 1)
influx = dlat * v + dlon * u
influx = influx.where(border_amazon)
print('resampling')
influx = influx.resample(time='1D').mean('time')
#
cpc_path = '~/phd_data/precip_1979a2017_CPC_AS.nc'
cpc = xr.open_dataarray(cpc_path)
cpc = cpc.rename({'lon': 'longitude', 'lat': 'latitude'})
cpc = cpc.assign_coords(longitude=(cpc.coords['longitude'].values + 180) % 360 - 180)

# --- Duration per grid point --- #


# ---- Fourier analysis ---- #

# basins = ['Tiete', 'Uruguai']
# basin = 'Uruguai'
# experiment = experiments[0]
# for basin in basins:
#     with open(outpath + experiment + 'config.txt') as file:  config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     # TODO: limitng number of files below WARNING
#
#     files = [f for f in glob.glob(outpath + experiment + "**/*_[0-9][0-9][0-9].nc", recursive=True)]
#     files1 = files[0:100]
#     files2 = files[100:200]
#     # files3 = files[200:300]
#     da1 = xr.open_mfdataset(files1)
#     da2 = xr.open_mfdataset(files2)
#     da = xr.concat([da1, da2], dim='time')
#     da = da.to_array()
#
#     da = da.where(da > 0, 1e-6)
#     da = np.log(np.sqrt(da)) / days
#     da = da.sortby('time').sel(time=slice(str(years.start), str(years.stop - 1))).resample(time='1D').mean('time')
#     mask = MAG[basin].interp(latitude=da.latitude, longitude=da.longitude, method='nearest')
#     da_ts = da.where(mask == 1).mean(['latitude', 'longitude'])
#     da_ts = da_ts.isel(variable=0)
#     cpc_ = cpc.sel(time=da_ts.time.values)
#     mask = MAG[basin].interp(latitude=cpc_.latitude, longitude=cpc_.longitude, method='nearest')
#     cpc_ts = cpc_.where(mask == 1).mean(['latitude', 'longitude'])
#
#     da_ts_ = da_ts.load()
#     f_cpc = xr.apply_ufunc(lambda x: (np.abs(np.fft.fft(x))), cpc_ts)
#     freqs = np.fft.fftfreq(cpc_ts.time.shape[0])
#     f_cpc = f_cpc.assign_coords(time=np.log(freqs**-1)).rename({'time': 'logperiod'}).sortby('logperiod')
#     f_cpc = f_cpc.where(f_cpc.logperiod > 0, drop=True  )
#     f_cpc = f_cpc.where(f_cpc.logperiod < 3.5, drop=True)
#     f_cpc_smoothed = f_cpc.rolling(logperiod=20).mean()
#     f_da_ts = xr.apply_ufunc(lambda x: (np.abs(np.fft.fft(x))), da_ts_, dask='allowed')
#     freqs = np.fft.fftfreq(da_ts_.time.shape[0])
#     f_da_ts = f_da_ts.assign_coords(time=np.log(freqs ** -1)).rename({'time': 'logperiod'}).sortby('logperiod')
#     f_da_ts = f_da_ts.where(f_da_ts.logperiod > 0, drop=True)
#     f_da_ts = f_da_ts.where(f_da_ts.logperiod < 3.5, drop=True)
#     f_da_ts_smoothed = f_da_ts.rolling(logperiod=20).mean()
#
#     fig, axs = plt.subplots(2, 1, figsize=[10, 10])
#     ax = axs[0]
#     ax2 = axs[1]
#
#     ax.set_xlabel('Log Period ')
#     ax2.set_xlabel('Log Period ')
#
#     ax2.set_ylabel('Lyapunov exponent power (1/day)')
#     ax.set_ylabel('Rainfall power (mm)')
#     secax = ax.secondary_xaxis('top', functions=(np.exp, np.log))
#     secax.set_xlabel('Period (days)')
#     secax.set_xticks(np.arange(1, 34, 2))
#     secax2 = ax2.secondary_xaxis('top', functions=(np.exp, np.log))
#     secax2.set_xlabel('Period (days)')
#     secax2.set_xticks(np.arange(1, 34, 2))
#
#     ax.plot(f_cpc.logperiod.values, f_cpc.values)
#     ax.plot(f_cpc.logperiod.values, f_cpc_smoothed.values)
#
#     ax2.plot(f_da_ts.logperiod.values, f_da_ts.values)
#     ax2.plot(f_da_ts.logperiod.values, f_da_ts_smoothed.values)
#
#     plt.savefig(f'tempfigs/fft_temp_{basin}.pdf')
#     plt.close()


# ---- Plotting ftle region composite ---- #
# print('Start plotting')
# import sys
# basins = [str(sys.argv[1])]
# experiment = experiments[0]
# for experiment in experiments:
#     with open(outpath + experiment + 'config.txt') as file:        config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})    raise Exception('Stop')
#
#     da = da.where(da > 0, 1e-6)
#     da = np.log(np.sqrt(da)) / days
#     da = da.sortby('time').sel(time=slice(str(years.start), str(years.stop - 1))).resample(time='1D').mean('time')
#     for basin in basins:
#
#         mask = MAG[basin].interp(latitude=da.latitude.values, longitude=da.longitude.values, method='nearest')
#         da_ts = da.where(mask == 1).mean(['latitude', 'longitude'])
#         threshold = da_ts.load().quantile(0.8)
#         da_avg_cz = da.where(da_ts > threshold, drop=True).mean('time') - \
#                     da.mean('time')
#
#         cpc_ = cpc.sel(time=da.time).interp(latitude=da.latitude.values, longitude=da.longitude.values)
#         cpc_ = cpc_.where(da_ts > threshold, drop=True) - cpc_.mean('time')
#         u_ = u.where(da_ts > threshold, drop=True).mean('time') - u.mean('time')
#         v_ = v.where(da_ts > threshold, drop=True).mean('time') - v.mean('time')
#
#         influx_anomaly_czs = influx.where(da_ts > threshold, drop=True) - influx.mean('time')
#         influx_ = influx_anomaly_czs.mean('time')
#         fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[20, 10])
#         cpc_.mean('time').plot(ax=axs[0], cmap='BrBG', vmin=-5, vmax=5, add_colorbar=False)
#         influx_.plot(ax=axs[0], cmap='seismic', vmin=-100, vmax=100, add_colorbar=False)
#         mask.plot.contour(levels=[0.9, 1.1], colors='black', ax=axs[0])
#         mask.plot.contour(levels=[0.9, 1.1], colors='black', ax=axs[1])
#         da_avg_cz.plot.contourf(cmap='RdBu', levels=21, ax=axs[1], add_colorbar=False)
#         axs[0].coastlines()
#         magnitude = (np.abs(u_.values) + np.abs(v_.values))
#         axs[0].streamplot(x=u_.longitude.values, y=u_.latitude.values,
#                       u=u_.values, v=v_.values, linewidth=2*magnitude/np.max(magnitude), color='k')
#         axs[0].set_xlim([da.longitude.min(), da.longitude.max()])
#         axs[0].set_ylim([da.latitude.min(), da.latitude.max()])
#         axs[1].coastlines()
#         axs[0].set_title('Precipitation and moisture flux anomalies')
#         axs[1].set_title(f'{days}day-FTLE anomaly during convergence in {basin}')
#
#         plt.savefig('tempfigs/FTLE_{days}_days_basin_{basin}.png'.format(days=str(int(days)), basin=basin))
#
#         plt.close()


# ---- Plotting correlation with south border ---- #
raise ValueError("!")
experiment = experiments[0]
with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
days = config['lcs_time_len'] / 4
files = [f for f in glob.glob(outpath + experiment + "**/*.nc", recursive=True)]
da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})
da = np.log(np.sqrt(da)) / days
da = da.resample(time='1D').mean()

cpc_ = cpc.sel(time=da.time).interp(latitude=da.latitude.values, longitude=da.longitude.values)

cpc_anomaly = xr.apply_ufunc(lambda x, y: x - y, cpc_.groupby('time.month'), cpc_.groupby('time.month').mean('time'))

south_border = border_amazon.where(border_amazon.latitude < -10, drop=True).where(border_amazon.longitude > -65, drop=True)
influx_south = influx.where(influx.latitude < -10, drop=True). \
     where(influx.longitude > -65, drop=True).where(influx != 0, drop=True).\
     mean(['latitude', 'longitude']).resample(time='1D').mean()
influx_anomaly = xr.apply_ufunc(lambda x, y: x - y, influx_south.groupby('time.month'),
                                influx_south.groupby('time.month').mean('time'), dask='allowed')

da_anomaly = xr.apply_ufunc(lambda x, y: x - y, da.groupby('time.month'),
                            da.groupby('time.month').mean('time'),
                            dask='allowed')
idx = common_index(influx_anomaly.time.values, da_anomaly.time.values)
correl = spearman_correlation(influx_anomaly.sel(time=idx), da_anomaly.sel(time=idx), dim='time')
correl = correl.where(~np.isnan(cpc_.mean('time')), drop=True)
n_events = len(idx)
t_corr = np.abs(correl*np.sqrt((n_events - 2)/(1-correl**2)))
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})

correl.plot.contourf(levels=21, cmap=cmr.fusion,ax=ax)
t_corr.plot.contourf(levels=[0, 2.57], ax=ax, hatches=['xxx', ' ', ' '], alpha=0, add_colorbar=False)
south_border.where(south_border==1).plot(cmap='black', levels=[0.9, 1.1], ax=ax, add_colorbar=False)
ax.coastlines()
ax.set_ylim([correl.latitude.min().values, correl.latitude.max().values])
ax.set_xlim([correl.longitude.min().values, correl.longitude.max().values])
plt.savefig('tempfigs/FTLE_correlation_flux.pdf')
plt.close()

ridges = xr.open_dataarray(outpath + experiment + 'ridges.nc')
ridges = ridges.where(da > 0.8, 0)
n_events = ridges.sum('time')
n_events = n_events.load()


ridges_flux = influx_anomaly * ridges
mean_ridges_flux = ridges_flux.mean('time')
mean_ridges_flux = mean_ridges_flux.load()
mean_ridges_flux.plot()
stdev = influx_anomaly.var('time') ** 0.5
t_test = n_events * mean_ridges_flux / stdev
t_test = np.abs(t_test)
t_test = t_test.load()

mean_ridges_flux.where(t_test > 5, np.nan).plot.contourf(levels=21, cmap=cmr.fusion)
t_test.plot.contourf(levels=[0, 2.57], ax=ax, hatches=['xxx', ' ', ' '], alpha=0, add_colorbar=False)


# correl = spearman_correlation(influx_anomaly.shift(time=0), cpc_anomaly, dim='time')
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[10, 10], )
#
# correl.where( ~np.isnan(cpc_anomaly.isel(time=0)) ).plot(ax=ax)
# south_border.where(south_border==1).plot(cmap='black', levels=[0.9, 1.1], ax=ax, add_colorbar=False)
# ax.coastlines()
# plt.savefig('tempfigs/precip_correlation_flux.pdf')
#
#
# print('Start plotting')
#


#
# experiment = experiments[0]
#
# for experiment in experiments:
#     print('!')
#     print(('Number of threads: ' + str(threading.active_count())))
#     with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})
#     da = da.sel(time=slice('1981', '1989'))
#     da = da.where(da > 0, 1e-6)
#     da = np.log(np.sqrt(da)) / days
#     da = da.resample(time='1D').mean('time')
#     threshold = 1
#     da = da.where(da > threshold, 0)
#     da = da.where(da == 0, 1)
#     da_var = da.groupby('time.season').var('time')
#     d_mean = da.groupby('time.season').mean('time')
#     d_mean_anomaly = xr.apply_ufunc(lambda x, y: x - y, da.groupby('time.season').mean('time'), da.mean('time'), dask='allowed')
#     fraction_of_days_anomaly_season = 100 * d_mean_anomaly
#     fraction_of_days_per_year = da.mean('time') * 100
#
#     stdev = da_var ** 0.5
#     t_test = (da.time.values.shape[0] ** 0.5) * d_mean_anomaly / stdev
#     t_test = np.abs(t_test)
#     t_99_percent_confidence = 2.581
#
#     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#     # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#     p = (fraction_of_days_anomaly_season).where(t_test > t_99_percent_confidence).plot.contourf(
#         subplot_kws={'projection': ccrs.PlateCarree()}, levels=11,
#         cmap='RdBu', vmin=-20, vmax=20, col='season', col_wrap=2)
#
#     for i, ax in enumerate(p.axes.flat): ax.coastlines()
#     plt.savefig('tempfigs/FTLE_pop_{days}_days_anomaly.pdf'.format(days=str(int(days))))
#     plt.close()
#     from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#     import matplotlib.ticker as mticker
#
#     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#     p = (fraction_of_days_per_year).plot.contourf(
#         ax=ax, levels=11,
#         cmap='Blues', vmin=0, vmax=50)
#     ax.coastlines(color='red')
#
#     gl = ax.gridlines(draw_labels=True)
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     gl.xlabels_top = False
#     gl.ylabels_right=False
#     gl.xlocator = mticker.FixedLocator([-80, -70,-60,-50,-40])
#     gl.xlines = False
#     gl.ylines = False
#     plt.savefig('tempfigs/FTLE_pop_{days}_days.pdf'.format(days=str(int(days))))
#     plt.close()

# #
# # # ---- Plotting ftle region composite ---- #
# time = 12
# for region in regions.keys():
#     for experiment in experiments:
#         with open(outpath + experiment + 'config.txt') as file:
#             config = eval(file.read())
#         days = config['lcs_time_len'] / 4
#
#
#         files = [f for f in glob.glob(outpath + experiment  + "**/*.nc", recursive=True)]
#         da = xr.open_mfdataset(files).to_array()
#         da = da.where(da > 0, 1e-6)
#         da = np.log(np.sqrt(da)) / days
#         da = da.sortby('time')
#         da = da.resample(time='1D').mean('time')
#         da_ts = latlonsel(da, **regions[region]).mean(['latitude', 'longitude']).isel(variable=0)
#         threshold = da_ts.load().quantile(0.8)
#         cpc_ = cpc.sel(time=da.time).interp(latitude=da.latitude.values, longitude=da.longitude.values)
#         cpc_ = cpc_.where(da_ts > threshold, drop=True) - cpc_.mean('time')
#         fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#         cpc_.mean('time', skipna=False).plot(ax=ax, cmap='BrBG',vmin=-5,vmax=5)
#         ax.coastlines()
#         ax.set_title('Time length: ' + str(days) + ' days' + ' - ' + pd.Timestamp(da.time.isel(time=time).values).strftime('%Y-%m-%d'))
#         plt.savefig('tempfigs/FTLE_{days}_days_{region}.pdf'.format(days=str(int(days)), region=region))
#         plt.close()
#

# ---- Plotting random time ftle ---- #
# for experiment in experiments:
#     with open(outpath + experiment + 'config.txt') as file:
#         config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#
#
#     files = [f for f in glob.glob(outpath + experiment  + "**/*.nc", recursive=True)]
#     da = xr.open_mfdataset(files).to_array()
#     da = np.log(np.sqrt(da)) / days
#     times = np.arange(0, 40, 4)
#     for time in times:
#
#         fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#         da.isel(variable=0, time=time).plot(ax=ax, cmap='inferno', vmin=0, vmax=2)
#         timestr = pd.Timestamp(da.time.isel(time=time).values).strftime('%Y-%m-%d')
#         print(timestr)
#         ax.coastlines()
#         ax.set_title('Time length: ' + str(days) + ' days' + ' - ' + timestr)
#         plt.savefig('tempfigs/FTLE_{days}_days_{timestr}.pdf'.format(days=str(int(days)), timestr=timestr))
#         plt.close()
#     #
# # ---- Plotting mean ftle ---- #
# for experiment in experiments:
#     print('!')
#     print(('Number of threads: ' + str(threading.active_count())))
#     with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     files = [f for f in glob.glob(outpath + experiment  + "**/*_[0-9][0-9][0-9].nc", recursive=True)]
#     files1 = files[0:50]
#     files2 = files[50:100]
#     da1 = xr.open_mfdataset(files1, combine='by_coords', chunks={'time': 100})
#     da1 = da1.chunk({'time': 100})
#     gc.collect()
#     da2 = xr.open_mfdataset(files2, combine='by_coords', chunks={'time': 100})
#     da2 = da2.chunk({'time': 100})
#     da = xr.concat([da1, da2], dim='time')
#     gc.collect()
#     print('!!')
#     print(('Number of threads: ' + str(threading.active_count())))
#     files = [f for f in glob.glob(outpath + experiment  + "**/*.nc", recursive=True)]
#     da = xr.open_mfdataset(files).to_array()
#     da = np.log(np.sqrt(da)) / days
#     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#     da.isel(variable=0).mean('time').plot(ax=ax, cmap='Blues', vmin=0, vmax=2)
#     ax.coastlines()
#     ax.set_title('Time length: ' + str(days) + ' days' + ' - ' + pd.Timestamp(da.time.isel(time=time).values).strftime('%Y-%m-%d'))
#     plt.savefig('tempfigs/FTLE_{}_days_mean.pdf'.format(str(int(days))))
#     plt.close()

# ---- Plotting corr with precip ---- #
# timeshifts = [0]
# experiment = experiments[0]
# for experiment in experiments:
#     for timeshift in timeshifts:
#         season = 1
#         with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
#         days = config['lcs_time_len'] / 4
#         da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})
#         da = da.sel(time=slice('1981', '1989'))
#         da = da.where(da > 0, 1e-6)
#         da = np.log(np.sqrt(da)) / days
#         da = da.resample(time='1D').mean('time')
#         season_mask = [(pd.Timestamp(x).month % 12 + 3) // 3 for x in da.time.values]
#         da['season_mask'] = ('time'), season_mask
#         da_ = da.where(da.season_mask==season, drop=True)
#         t_95_percent_confidence = 2.576
#
#         cpc_ = cpc.sel(time=da_.time).interp(latitude=da_.latitude.values, longitude=da_.longitude.values, method='linear')
#         cpc_ = cpc_.rolling(time=int(days)).mean()
#         n_events = cpc_.where(cpc_ > 1, 0)
#         n_events = n_events.where(n_events <= 1, 1)
#         n_events = n_events.sum('time')
#
#         da_ = da_.where(n_events > 3, np.nan)
#         cpc_ = cpc_.where(n_events > 3, np.nan)
#         n_events = n_events.where(n_events > 3, np.nan)
#         corr = spearman_correlation(cpc_, da_, 'time')
#         corr = corr.where(~np.isnan(cpc_.mean('time')), drop=True)
#         t_corr = corr*np.sqrt((n_events - 2)/(1-corr**2))
#         fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#         corr.plot.contourf(levels=21, ax=ax, cmap='YlGnBu', vmin=0, vmax=0.6)
#         t_corr.plot.contourf(levels=[0, t_95_percent_confidence], ax=ax, hatches=['xxx', ' ', ' '],alpha=0, add_colorbar=False)
#
#         ax.coastlines()
#         ax.set_title('Time length: ' + str(days) + ' days')
#         plt.savefig('tempfigs/corr_{day}_days_season_{season}.pdf'.format(
#             season=season, day=str(int(days))))
#         plt.close()

# cpc_ = cpc_.differentiate('time', datetime_unit='1D')
# corr = spearman_correlation(cpc_, da.where(np.abs(cpc_) > 0, 0), 'time')
# corr = corr.where(~np.isnan(cpc_.isel(time=-1)), drop=True)
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
# corr.plot(ax=ax, cmap='jet',
#                                                              vmin=0.1, vmax=0.4)
# ax.coastlines()
# ax.set_title('Time length: ' + str(days) + ' days')
# plt.savefig('tempfigs/corr_diff_1_{day}_days.pdf'.format(day=str(int(days))))
# plt.close()
#
# cpc_ = cpc_.differentiate('time', datetime_unit='1D')
# corr = spearman_correlation(cpc_, da.where(np.abs(cpc_) > 0, 0), 'time')
# corr = corr.where(~np.isnan(cpc_.isel(time=-1)), drop=True)
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
# corr.plot(ax=ax, cmap='jet',
#                                                              vmin=0.1, vmax=0.4)
# ax.coastlines()
# ax.set_title('Time length: ' + str(days) + ' days')
# plt.savefig('tempfigs/corr_diff_2_{day}_days_season_{season}.pdf'.format(season=season,
#                                                                          day=str(int(days))))
# plt.close()
# # #
# # for time in da.time.values:
# #     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
# #
# #     da.sel(time=time).isel(variable=0).plot.contourf(levels=20,cmap='nipy_spectral', vmin=0, vmax=7, ax=ax, transform=ccrs.PlateCarree())
# #     ax.coastlines()
# #
# #     plt.show()
