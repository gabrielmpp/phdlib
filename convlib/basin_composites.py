import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import bottleneck
import pandas as pd
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
        z = xr.open_dataset(datapath + zfilename.format(year=year), chunks={'time': 100})
        u = xr.open_dataset(datapath + ufilename.format(year=year), chunks={'time': 100})
        v = xr.open_dataset(datapath + vfilename.format(year=year), chunks={'time': 100})
        z = z.assign_coords(longitude=(z.coords['longitude'].values + 180) % 360 - 180)
        u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
        v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
        u = u.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        z = z.sel(latitude=slice(20, -70), longitude=slice(-160, -10))
        v = v.sel(latitude=slice(20, -60), longitude=slice(-80, -10))

        z = z.to_array().isel(variable=0).drop('variable')
        u = u.to_array().isel(variable=0).drop('variable')
        v = v.to_array().isel(variable=0).drop('variable')
        u_list.append(u)
        v_list.append(v)
        z_list.append(z)
        print(str(year))
    u = xr.concat(u_list, dim='time')
    v = xr.concat(v_list, dim='time')
    z = xr.concat(z_list, dim='time')
    [x.close() for x in u_list]
    [x.close() for x in v_list]
    [x.close() for x in z_list]
    u_list = None
    v_list = None
    z_list = None
    return u, v, z


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian


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

years = slice(1981, 1989)
datapath = '/gws/nopw/j04/primavera1/observations/ERA5/'
vfilename = 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
ufilename = 'viwve_ERA5_6hr_{year}010100-{year}123118.nc'
zfilename = 'zg_250_ERA5_6hrPlevPt_{year}010100-{year}123118.nc'

u, v, z = read_u_v(years)
print('1')
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

cpc_path = '~/phd_data/precip_1979a2017_CPC_AS.nc'
cpc = xr.open_dataarray(cpc_path)
cpc = cpc.rename({'lon': 'longitude', 'lat': 'latitude'})
cpc = cpc.assign_coords(longitude=(cpc.coords['longitude'].values + 180) % 360 - 180)

print('2')
import sys

basins = [str(sys.argv[1])]
experiment = experiments[0]

for experiment in experiments:

    with open(outpath + experiment + 'config.txt') as file:
        config = eval(file.read())
    days = config['lcs_time_len'] / 4
    da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})
    da = da.sel(time=slice('1981', '1989'))
    da = da.where(da > 0, 1e-6)
    da = np.log(np.sqrt(da)) / days
    da = da.sortby('time').sel(time=slice(str(years.start), str(years.stop - 1))).resample(time='1D').mean('time')
    da.compute()
    plt.style.use('bmh')
    season_mask = [(pd.Timestamp(x).month % 12 + 3) // 3 for x in da.time.values]
    da['season_mask'] = ('time'), season_mask
    # ridges = xr.open_dataarray(outpath + experiment + 'ridges.nc')
    # da = da * ridges.sel(time=da.time)

    for basin in basins:
        season = 1
        print('3')
        mask = MAG[basin].interp(latitude=da.latitude.values, longitude=da.longitude.values, method='nearest')
        da_season = da.where(da.season_mask == season, drop=True)
        da_ts = da_season.where(mask == 1).mean(['latitude', 'longitude'])
        threshold = 0.8
        da_ts = da_ts.where(da_ts > threshold, 0)
        da_ts = da_ts.where(da_ts < threshold, 1)
        raise ValueError('!')
        da_avg_cz = da.where(da_ts == 1, drop=True).mean('time') - \
                    da.sel(time=da_ts.time).mean('time')

        stdev_cz = da.var('time') ** 0.5
        t_test_cz = (da_ts.time.values.shape[0] ** 0.5) * da_avg_cz / stdev_cz
        t_test_cz = np.abs(t_test_cz)
        #
        #
        # # ---- Plotting geopotential anomaly ---- #
        z_ = z.sel(time=da.time)
        z_['season_mask'] = ('time'), season_mask
        z_ = z_.where(z_.season_mask == season, drop=True)
        z_anomaly = z_.where(da_ts == 1, drop=True).mean('time') - z_.mean('time')
        stdev = z_.var('time') ** 0.5
        t_test = (da_ts.time.values.shape[0] ** 0.5) * z_anomaly / stdev
        t_test = np.abs(t_test)
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        t_99_percent_confidence = 2.5
        z_anomaly.where(t_test > t_99_percent_confidence).plot.contourf(ax=ax, cmap='BrBG', levels=11,
                                                                        transform=ccrs.PlateCarree(),
                                                                        vmin=-500,
                                                                        vmax=500)
        mask.plot.contour(levels=[0.9, 1.1], colors='gray', ax=ax)

        ax.coastlines()
        plt.savefig(f'tempfigs/geopotential_anomaly_{basin}_season_{season}.pdf')
        plt.close()

        # ---- Plotting geopotential anomaly lagged 3 days ---- #

        z_ = z.sel(time=da.time)

        z_['season_mask'] = ('time'), season_mask
        z_ = z_.where(z_.season_mask == season, drop=True)
        z_anomaly = z_.shift(time=3).where(da_ts == 1, drop=True).mean('time') - z_.mean('time')
        stdev = z.var('time') ** 0.5
        t_test = (da_ts.time.values.shape[0] ** 0.5) * z_anomaly / stdev
        t_test = np.abs(t_test)
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
        z_anomaly.where(t_test > t_99_percent_confidence).plot.contourf(ax=ax, cmap='BrBG', levels=11,
                                                                        transform=ccrs.PlateCarree(),
                                                                        vmin=-500,
                                                                        vmax=500)
        mask.plot.contour(levels=[0.9, 1.1], colors='gray', ax=ax)

        ax.coastlines()
        plt.savefig(f'tempfigs/geopotential_anomaly_{basin}_season_{season}_lag3.pdf')
        plt.close()
        #
        # # ---- Plotting rainfall and flux means ---- #

        cpc_ = cpc.sel(time=da.time)
        cpc_['season_mask'] = ('time'), season_mask
        cpc_ = cpc_.where(cpc_.season_mask == season, drop=True).mean('time')

        u_ = u.sel(time=da_ts.time)
        v_ = v.sel(time=da_ts.time)
        u_ = u_.load()
        v_ = v_.load()

        u_ = u_.mean('time')

        v_ = v_.mean('time')
        import seaborn as sns

        influx_ = influx.sel(time=da_ts.time)
        influx_ = influx_.mean('time')
        influx_ = influx_.load()

        fig, axs = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[10, 10])
        cpc_.plot.contourf(levels=16, ax=axs, cmap=sns.cubehelix_palette(dark=0, light=1, as_cmap=True),
                           vmin=0, vmax=15, add_colorbar=True)
        influx_.plot(ax=axs, cmap='seismic', vmin=-100, vmax=100, add_colorbar=True)

        axs.coastlines()
        magnitude = (np.abs(u_.values) + np.abs(v_.values))
        axs.streamplot(x=u_.longitude.values, y=u_.latitude.values,
                       u=u_.values, v=v_.values, linewidth=2 * magnitude / np.max(magnitude), color='k')
        axs.set_xlim([da.longitude.min(), da.longitude.max()])
        axs.set_ylim([da.latitude.min(), da.latitude.max()])
        axs.set_title('Precipitation and moisture flux')

        plt.savefig('tempfigs/FTLE_{days}_days_basin_{basin}_season_{season}.pdf'.format(
            days=str(int(days)), basin=basin, season=season))

        plt.close()

        # # ---- Plotting rainfall and flux anomaly ---- #

        cpc_ = cpc.sel(time=da.time)
        cpc_['season_mask'] = ('time'), season_mask
        cpc_ = cpc_.where(cpc_.season_mask == season, drop=True)
        stdev_cpc = cpc_.var('time') ** 0.5
        cpc_ = cpc_.where(da_ts == 1, drop=True).mean('time') - cpc_.mean('time')
        t_test_cpc = (da_ts.time.values.shape[0] ** 0.5) * cpc_ / stdev_cpc
        t_test_cpc = np.abs(t_test_cpc)
        u_ = u.sel(time=da_ts.time)
        v_ = v.sel(time=da_ts.time)
        u_ = u_.load()
        v_ = v_.load()
        u_ = u_.where(da_ts == 1, drop=True).mean('time') - u_.mean('time')

        v_ = v_.where(da_ts == 1, drop=True).mean('time') - v_.mean('time')
        influx_ = influx.sel(time=da_ts.time)
        stdev_influx = influx_.var('time') ** 0.5
        influx_anomaly_czs = influx_.where(da_ts == 1, drop=True) - influx_.mean('time')
        influx_ = influx_anomaly_czs.mean('time')
        influx_ = influx_.load()

        t_test_influx = (da_ts.time.values.shape[0] ** 0.5) * influx_ / stdev_influx
        t_test_influx = np.abs(t_test_influx)
        cpc_toplot = cpc_.where(t_test_cpc > t_99_percent_confidence).load()
        influx_toplot = influx_.where(t_test_influx > t_99_percent_confidence).load()
        da_toplot = da_avg_cz.where(t_test_cz > t_99_percent_confidence).load()
        magnitude = (np.abs(u_.values) + np.abs(v_.values))

        fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()})
        cpc_.name = 'Rainfall anomaly (mm/day)'
        da_avg_cz.name = 'FTLE anomaly (1/day)'
        cmap = cmr.iceburn_r
        cpc_toplot.plot.contourf(levels=21, ax=axs[0], cmap=cmap, vmin=-5, vmax=5,
                                 add_colorbar=True,
                                 cbar_kwargs={'shrink': 0.6})

        p = influx_toplot.plot(ax=axs[0], cmap='seismic',
                               vmin=-100, vmax=100,
                               add_colorbar=False)

        plt.colorbar(p, ax=axs[0], orientation='horizontal', shrink=0.9)
        mask.plot.contour(levels=[0.9, 1.1], colors='gray', ax=axs[0], linewidths=0.9)
        mask.plot.contour(levels=[0.9, 1.1], colors='gray', ax=axs[1], linewidths=0.9)
        da_toplot.plot.contourf(
            cmap=cmr.redshift, levels=21, ax=axs[1], add_colorbar=True, cbar_kwargs={'shrink': 0.6})
        plt.colorbar(p, ax=axs[1], orientation='horizontal', shrink=0.9)
        axs[0].coastlines()
        axs[0].streamplot(x=u_.longitude.values, y=u_.latitude.values, arrowsize=0.6,
                          u=u_.values, v=v_.values, linewidth=magnitude / np.max(magnitude), color='k')
        axs[0].set_xlim([da.longitude.min(), da.longitude.max()])
        axs[0].set_ylim([da.latitude.min(), da.latitude.max()])
        axs[1].coastlines()
        # axs[0].set_title('Precipitation and moisture flux anomalies')
        # axs[1].set_title(f'{days}day-FTLE anomaly during convergence in {basin}')

        plt.savefig('tempfigs/FTLE_{days}_days_basin_{basin}_season_{season}_anomalies.pdf'.format(
            days=str(int(days)), basin=basin, season=season))

        plt.close()
        print('Done !')

        # ---- Plotting duration histograms ---- #
        # from scipy.signal import find_peaks, peak_widths
        # mask = MAG[basin].interp(latitude=cpc.latitude.values,
        #                          longitude=cpc.longitude.values, method='nearest')
        # cpc_ts = cpc.where(mask==1, drop=True).mean(['latitude', 'longitude'])
        # cpc_ts = cpc_ts.where(cpc_ts > 5, 0)
        # cpc_ts = cpc_ts.where(cpc_ts < 5, 1)
        # cpc_ts_cz = cpc_ts.where(da_ts==1, drop=True)
        #
        # peaks = find_peaks(cpc_ts.values)[0]
        # peaks_cz = find_peaks(cpc_ts_cz.values)[0]
        # peaks_w = peak_widths(cpc_ts, peaks, rel_height=0.5)[0]
        # peaks_w_cz = peak_widths(cpc_ts_cz, peaks_cz, rel_height=0.5)[0]
        # fig, axs = plt.subplots(2, 1)
        # axs[0].hist(peaks_w_cz, range=[1, 10])
        # axs[0].set_xlabel('Duration of rainfall events (day)')
        # axs[0].set_ylabel('Count')
        # axs[1].set_xlabel('Duration of rainfall events (day)')
        # axs[1].set_ylabel('Count')
        # axs[1].hist(peaks_w, range=[1, 10])
        # plt.savefig(f'tempfigs/hist_duration_{basin}.pdf')
        #
