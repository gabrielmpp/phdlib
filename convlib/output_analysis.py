import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
import dask
from multiprocessing.pool import ThreadPool
dask.config.set(scheduler='threads') # Global config
dask.config.set(pool=ThreadPool(80)) # To avoid omp error
import gc
import threading
import glob
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import bottleneck
from scipy import stats
from xr_tools.tools import latlonsel
from scipy.signal import savgol_filter
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




# outpath = '/home/gab/phd/data/FTLE_ERA5/'
outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
experiments = ['experiment_cfc512b7-f65b-480b-9935-91349b9f2e75/',
               'experiment_d19c89e8-69a7-4a98-8022-9462c7d1dab1/',
               'experiment_6f659732-e804-474c-a060-eda671a7c204/',
               'experiment_e26c9242-2537-4fb8-a5e3-950f0e55c5c8/']

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

years = slice(2001, 2005)
datapath = '/gws/nopw/j04/primavera1/observations/ERA5/'
# vfilename = 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
# ufilename = 'viwve_ERA5_6hr_{year}010100-{year}123118.nc'
# u_list, v_list = [], []
# for year in np.arange(years.start, years.stop):
#     u = xr.open_dataset(datapath + ufilename.format(year=year))
#     v = xr.open_dataset(datapath + vfilename.format(year=year))
#     u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
#     v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
#     u = u.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
#     v = v.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
#
#     u = u.to_array().isel(variable=0).drop('variable')
#     v = v.to_array().isel(variable=0).drop('variable')
#     u_list.append(u)
#     v_list.append(v)
#     print(str(year))
# u = xr.concat(u_list, dim='time')
# v = xr.concat(v_list, dim='time')
#
MAG = xr.open_dataset('~/phdscripts/phdlib/convlib/data/xarray_mair_grid_basins.nc')
MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})
# dlat = np.sign(MAG['amazon'].diff('latitude'))
# dlon = np.sign(MAG['amazon'].diff('longitude'))
# border_amazon = np.abs(dlat) + np.abs(dlon)
# border_amazon = border_amazon.where(border_amazon <= 1, 1)
# influx = dlat * v + dlon * u
# influx = influx.where(border_amazon)
# print('resampling')
# influx = influx.resample(time='1D').mean('time')

cpc_path = '~/phd_data/precip_1979a2017_CPC_AS.nc'
cpc = xr.open_dataarray(cpc_path)
cpc = cpc.rename({'lon': 'longitude', 'lat': 'latitude'})
cpc = cpc.assign_coords(longitude=(cpc.coords['longitude'].values + 180) % 360 - 180)

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
# basins = ['Tiete', 'Doce', 'Uruguai']
# basin='Tiete'
# experiment=experiments[0]
# for experiment in experiments:
#     with open(outpath + experiment + 'config.txt') as file:        config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     files = [f for f in glob.glob(outpath + experiment + "**/*.nc", recursive=True)]
#     files1 = files[0:100]
#     files2 = files[100:200]
#
#     da1 = xr.open_mfdataset(files1)
#     da2 = xr.open_mfdataset(files2)
#     da = xr.concat([da1, da2], dim='time')
#     da = da.where(da > 0, 1e-6)
#     da = np.log(np.sqrt(da)) / days
#     da = da.sortby('time').sel(time=slice(str(years.start), str(years.stop - 1))).resample(time='1D').mean('time')
#     for basin in basins:
#
#         mask = MAG[basin].interp(latitude=da.latitude, longitude=da.longitude, method='nearest')
#         da_ts = da.where(mask == 1).mean(['latitude', 'longitude'])
#         threshold = da_ts.load().quantile(0.8)
#         da_avg_cz = da.where(da_ts > threshold, drop=True).mean('time') - \
#                     da.mean('time')
#
#         cpc_ = cpc.sel(time=da.time).interp(latitude=da.latitude.values, longitude=da.longitude.values)
#         cpc_ = cpc_.where(da_ts > threshold, drop=True) - cpc_.mean('time')
#         u_ = u.where(da_ts > threshold, drop=True).mean('time') - u.mean('time')
#         v_ = v.where(da_ts > threshold, drop=True).mean('time') - u.mean('time')
#
#         influx_anomaly_czs = influx.where(da_ts > threshold, drop=True) - influx.mean('time')
#         influx_ = influx_anomaly_czs.mean('time')
#         fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[20, 10])
#         cpc_.to_array().isel(variable=0).mean('time').plot(ax=axs[0], cmap='BrBG', vmin=-5, vmax=5)
#         influx_.to_array().isel(variable=0).plot(ax=axs[0], cmap='seismic', vmin=-100, vmax=100)
#         mask.plot.contour(levels=[0.9, 1.1], colors='black', ax=axs[0])
#         mask.plot.contour(levels=[0.9, 1.1], colors='black', ax=axs[1])
#         da_avg_cz.to_array().isel(variable=0).plot.contourf(cmap='RdBu', levels=21, ax=axs[1])
#         axs[0].coastlines()
#         axs[0].quiver(X=u_.longitude.values, Y=u_.latitude.values,
#                       U=u_.values, v=v_.values)
#         axs[0].set_xlim([da.longitude.min, da.longitude.max])
#         axs[0].set_ylim([da.latitude.min, da.latitude.max])
#         axs[1].coastlines()
#         axs[0].set_title('Precipitation and moisture flux anomalies')
#         axs[1].set_title(f'{days}day-FTLE anomaly during convergence in {basin}')
#
#         plt.savefig('tempfigs/FTLE_{days}_days_basin_{basin}.pdf'.format(days=str(int(days)), basin=basin))
#
#         plt.close()

# cpc_ = cpc.sel(time=da.time).interp(latitude=da.latitude.values, longitude=da.longitude.values)
#
# cpc_anomaly = xr.apply_ufunc(lambda x, y: x - y, cpc_.groupby('time.month'), cpc_.groupby('time.month').mean('time'))
#
# south_border = border_amazon.where(border_amazon.latitude < -10, drop=True).where(border_amazon.longitude > -65, drop=True)
# influx_south = influx.where(influx.latitude < -10, drop=True). \
#      where(influx.longitude > -65, drop=True).where(influx != 0, drop=True).\
#      mean(['latitude', 'longitude']).resample(time='1D').mean()
# influx_anomaly = xr.apply_ufunc(lambda x, y: x - y, influx_south.groupby('time.month'), influx_south.groupby('time.month').mean('time'))
#
# da_anomaly = xr.apply_ufunc(lambda x, y: x - y, da.groupby('time.month'),
#                             da.groupby('time.month').mean('time'),
#                             dask='allowed')
#
# correl = spearman_correlation(influx_anomaly, da_anomaly, dim='time')
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[10, 10], )
#
# correl.plot(ax=ax)
# south_border.where(south_border==1).plot(cmap='black', levels=[0.9, 1.1], ax=ax, add_colorbar=False)
# ax.coastlines()
# plt.savefig('tempfigs/FTLE_correlation_flux.pdf')
#
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


def safely_read_multiple_files(files, size_of_chunk=20, concat_dim = 'time'):

    import gc
    import threading
    try:
        import psutil
    except:
        psutil_available = False
    else:
        psutil_available = True


    print(('Number of active threads: ' + str(threading.active_count())))
    if psutil_available:
        p = psutil.Process()
        print('Process using ' + str(round(p.memory_percent(), 2)) + '% of the total system memory.')

    array_list = []
    chunked_array_list = []
    for i, file in enumerate(files):
        print('Reading file ' + str(i))
        array_list.append(xr.open_dataarray(file))
        if i % size_of_chunk == 0:
            if psutil_available:
                print('Process using ' + str(round(p.memory_percent(), 2)) + '% of the total system memory.')
            print(('Number of active threads: ' + str(threading.active_count())))
            chunked_array_list.append(
                xr.concat(array_list, dim=concat_dim)
            )
            [file.close() for file in array_list]  # closing all handles
            array_list = []  #  resetting array
            gc.collect()

    return xr.concat(chunked_array_list, dim='time')



#
#
# experiment = experiments[0]
#
# for experiment in experiments:
#     print('!')
#     print(('Number of threads: ' + str(threading.active_count())))
#     with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
#     days = config['lcs_time_len'] / 4
#     files = [f for f in glob.glob(outpath + experiment + "**/*_[0-9][0-9][0-9].nc", recursive=True)]
#
#     da = safely_read_multiple_files(files[0:300])
#
#     da = da.where(da > 0, 1e-6)
#     da = np.log(np.sqrt(da)) / days
#     da = da.sortby('time').sel(time=slice(str(years.start), str(years.stop - 1))).resample(time='1D').mean('time')
#     threshold = 2 / days
#     da = da.where(da > threshold, 0)
#     da = da.where(da == 0, 1)
#     da_var = da.groupby('time.season').var('time')
#     d_mean = da.groupby('time.season').mean('time')
#     d_mean_anomaly = xr.apply_ufunc(lambda x, y: x - y, da.groupby('time.season').mean('time'), da.mean('time'), dask='allowed')
#
#     # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#     p = (d_mean_anomaly*360/4).plot.contourf(
#         subplot_kws={'projection': ccrs.PlateCarree()}, levels=11,
#         cmap='BrBG', vmin=-15, vmax=15, col='season', col_wrap=2)
#
#     for i, ax in enumerate(p.axes.flat): ax.coastlines()
#     plt.savefig('tempfigs/FTLE_pop_{days}_days_anomaly.pdf'.format(days=str(int(days))))
#     plt.close()
#     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
#     p = (d_mean.mean('season')*360/4).plot.contourf(
#         ax=ax, levels=11,
#         cmap='viridis', vmin=0, vmax=50)
#     ax.coastlines()
#
#     plt.savefig('tempfigs/FTLE_pop_{days}_days.pdf'.format(days=str(int(days))))
#     plt.close()
#
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

# # ---- Plotting corr with precip ---- #
timeshifts = [0]
for experiment in experiments:
    for timeshift in timeshifts:
        print('!')
        print(('Number of threads: ' + str(threading.active_count())))
        with open(outpath + experiment + 'config.txt') as file: config = eval(file.read())
        days = config['lcs_time_len'] / 4


        files = [f for f in glob.glob(outpath + experiment  + "**/*_[0-9][0-9][0-9].nc", recursive=True)]

        da = safely_read_multiple_files(files[0:300])

        print('!!')
        print(('Number of threads: ' + str(threading.active_count())))

        da = np.log(np.sqrt(da)) / days
        da = da.sortby('time')

        da = da.resample(time='1D').mean('time')
        cpc_ = cpc.sel(time=da.time).interp(latitude=da.latitude.values, longitude=da.longitude.values, method='linear')
        cpc_ = cpc_.rolling(time=int(days)).mean()
        corr = spearman_correlation(cpc_, da.where(cpc_ > 0.1, 0), 'time')
        corr = corr.where(~np.isnan(cpc_.isel(time=-1)), drop=True)

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        corr.plot( ax=ax, cmap='jet',
                                                                     vmin=0.1, vmax=1)
        ax.coastlines()
        ax.set_title('Time length: ' + str(days) + ' days')
        plt.savefig('tempfigs/corr_{day}_days.pdf'.format(day=str(int(days))))
        plt.close()

        cpc_ = cpc_.differentiate('time', datetime_unit='1D')
        corr = spearman_correlation(cpc_, da.where(np.abs(cpc_) > 0, 0), 'time')
        corr = corr.where(~np.isnan(cpc_.isel(time=-1)), drop=True)

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        corr.plot(ax=ax, cmap='jet',
                                                                     vmin=0.1, vmax=0.4)
        ax.coastlines()
        ax.set_title('Time length: ' + str(days) + ' days')
        plt.savefig('tempfigs/corr_diff_1_{day}_days.pdf'.format(day=str(int(days))))
        plt.close()

        cpc_ = cpc_.differentiate('time', datetime_unit='1D')
        corr = spearman_correlation(cpc_, da.where(np.abs(cpc_) > 0, 0), 'time')
        corr = corr.where(~np.isnan(cpc_.isel(time=-1)), drop=True)

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        corr.plot(ax=ax, cmap='jet',
                                                                     vmin=0.1, vmax=0.4)
        ax.coastlines()
        ax.set_title('Time length: ' + str(days) + ' days')
        plt.savefig('tempfigs/corr_diff_2_{day}_days.pdf'.format(day=str(int(days))))
        plt.close()
# # #
# # for time in da.time.values:
# #     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
# #
# #     da.sel(time=time).isel(variable=0).plot.contourf(levels=20,cmap='nipy_spectral', vmin=0, vmax=7, ax=ax, transform=ccrs.PlateCarree())
# #     ax.coastlines()
# #
# #     plt.show()
