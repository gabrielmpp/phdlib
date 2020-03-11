import scipy.signal as signal
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from phdlib.utils.xrumap import autoencoder as xru

import cartopy.crs as ccrs
import cartopy.feature as cfeature

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
def latlonsel(array, array_like=None, lat=None, lon=None, latname='lat', lonname='lon'):
    """
    Function to crop array based on lat and lon intervals given by slice or list.
    This function is able to crop across cyclic boundaries.

    :param array: xarray.Datarray
    :param lat: list or slice (min, max)
    :param lon: list or slice(min, max)
    :return: cropped array
    """
    assert latname in array.coords, f"Coord. {latname} not present in array"
    assert lonname in array.coords, f"Coord. {lonname} not present in array"

    if isinstance(array_like, xr.DataArray):
        lat1 = array[latname].min().values
        lat2 = array[latname].max().values
        lon1 = array[lonname].min().values
        lon2 = array[lonname].max().values
    elif isinstance(lat, slice):
        lat1 = lat.start
        lat2 = lat.stop
    elif isinstance(lat, list):
        lat1 = lat[0]
        lat2 = lat[1]
    if isinstance(lon, slice):
        lon1 = lon.start
        lon2 = lon.stop
    elif isinstance(lon, list):
        lon1 = lat[0]
        lon2 = lat[1]

    lonmask = (array[lonname] < lon2) & (array[lonname] > lon1)
    latmask = (array[latname] < lat2) & (array[latname] > lat1)
    array = array.where(lonmask, drop=True)
    array = array.where(latmask, drop=True)
    return array


def model_analysis():
    ds = xr.open_dataset('data/ds.nc')

    array = ds.P.resample(time='1D').sum()
    print(array)
    array = array.stack(dict(stacked=['coupling', 'A_amazon']))
    # array = dict(array.groupby('time.season'))['DJF']
    groups = list(array.groupby('stacked'))
    power_list = []
    for label, group in groups:
        freq, power = signal.periodogram(group.values, axis=0)
        power_list.append(xr.DataArray(power, dims=['freq'], coords=dict(freq=freq)))

    array = xr.concat(power_list, dim=array.stacked)
    array = array.unstack()
    array = xr.apply_ufunc(lambda x: np.log(x), array)
    array = array.assign_coords(freq=1 / array.freq).rename(dict(freq='period'))
    # array = array.coarsen(period=Z, boundary='trim').mean()
    array.sel(A_amazon=10, period=slice(15, 2)).plot.contourf(levels=30, vmin=-26, vmax=-20, cmap='plasma')
    plt.show()
    array.sel(A_amazon=7, period=slice(15, 2)).plot.contourf(levels=30, vmin=-26, vmax=-20, cmap='plasma')
    plt.show()
    (array.sel(A_amazon=10, period=slice(15, 2)) - array.sel(A_amazon=7, period=slice(15, 2))).plot.contourf(levels=30,
                                                                                                             vmin=-3.5,
                                                                                                             vmax=3.5,
                                                                                                             cmap='RdBu')
    plt.show()
    array.sel(A_amazon=10, period=4, method='nearest').plot.line(x='coupling')
    plt.show()

    # qqplot

    ds = xr.open_dataset('data/ds.nc')
    idxs = np.array([pd.to_datetime(t).month in [1, 2, 12] for t in ds.time.values])
    ds = ds.sel(time=idxs)
    array = ds.P.resample(time='1D').sum() * 86400
    arrayQ = ds.Q.resample(time='1D').sum()  # * 86400
    array_CZ = array.where(arrayQ > arrayQ.quantile(0.8), drop=True)
    quants = np.arange(0, 1, 0.02)
    quant_cz = []
    quant_nocz = []
    for quant in quants:
        quant_cz.append(array_CZ.sel(coupling=4).quantile(quant).values)
        quant_nocz.append(array.sel(coupling=4).quantile(quant).values)

    plt.style.use('seaborn')
    plt.plot([0, 8], [0, 8], color='black', linestyle='dashed')

    plt.scatter(y=quant_cz, x=quant_nocz)
    plt.show()


def assert_time_is_last_entry(da):
    dims = list(da.dims)
    dims.remove('time')
    dims.append('time')
    return da.transpose(*dims)


def hilbert_spectrum(emd_array, out_type='period', apply_gauss=False):
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter
    def run_hilbert(group):
        analytic_signal = hilbert(group)
        phase = np.unwrap(np.angle(analytic_signal))
        #  freq = 4 * np.diff(phase) / (2 * np.pi)
        return phase

    # dask_array = emd_array.chunk(dict(encoded_dims=1, lat=1, lon=1))
    # temp_array = dask_array.stack({'stacked': ['encoded_dims', 'lat', 'lon']})
    emd_array = assert_time_is_last_entry(emd_array)
    phase = xr.apply_ufunc(lambda x: run_hilbert(x),
                           emd_array,
                           )
    freq = 4 * phase.diff('time') / (2 * np.pi)
    freq = freq.unstack().load()

    if out_type == 'freq':
        return freq
    elif out_type == 'period':
        # periods_array = np.linspace(20, 4, 30)
        # periods_array = np.array([30, 25, 20, 15, 10, 5, 1])
        periods_array = np.array([90 ,30 ,15, 8,4, 2, 0.5])
        k = 0
        periods = []

        while k < (periods_array.shape[0] - 1):  # TODO make it faster
            mask = (freq > 1 / periods_array[k]) & (freq < 1 / periods_array[k + 1])
            periods.append(emd_array.isel(time=slice(None, -1)).where(mask).sum('encoded_dims'))
            k += 1
            print(k)

        from scipy.ndimage import gaussian_filter

        array = xr.concat(periods, dim=pd.Index(periods_array[:-1], name='Period'))
        return array
    else:
        raise ValueError(f'out_type {out_type} not supported.')


def plot_periods(periods):

    # periods = periods.resample(time='M').sum('time')


    # periods_list = [5, 10, 15, 20, 25]
    periods_list=[90 ,30 ,15, 8,4, 2]

    for time in periods.time.values:
        p = periods.sel(Period=periods_list, method='nearest').sel(time=time).plot.contourf(cmap='RdBu', vmax=10, vmin=-10,
                                                                                          transform=ccrs.PlateCarree(),
                                                                                          col='Period', col_wrap=3,
                                                                                          aspect=periods.lon.shape[0] /
                                                                                                 periods.lat.shape[0],
                                                                                          levels=50, subplot_kws={
                'projection': ccrs.PlateCarree()})
        for i, ax in enumerate(p.axes.flat):
            fraction_of_power   = np.abs(periods.sel(Period=periods_list[i],
                                            method='nearest').sel(time=time)).sum(['lat', 'lon']).values / \
                np.abs(periods.sel(time=time)).sum(['Period', 'lat', 'lon']).values
            ax.text(-44, -26, str(round(fraction_of_power*100))+'%')
            ax.coastlines()
            ax.add_feature(states_provinces, edgecolor='gray')
        plt.suptitle(pd.Timestamp(time).strftime("%Y-%m-%d"))
        plt.savefig(f'figs/plot_{time}.png')
        plt.close()


sp = dict(lat=-23, lon=-45)

spc = xr.open_dataarray('~/prec_emd_for_mair_unscaled.nc')
# for time in spc.time.values[:100]:
#     f, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
#     spc.sel(time=time).sum('encoded_dims').plot.contourf(levels=30, vmin=0, vmax=50,
#                                        transform=ccrs.PlateCarree(),
#                                        ax=ax)
#
#     ax.coastlines()
#     ax.add_feature(cfeature.BORDERS)
#     ax.add_feature(states_provinces, edgecolor='black')
#     ax.set_extent([spc.lon.min().values, spc.lon.max().values, spc.lat.min().values, spc.lat.max().values])
#     plt.suptitle(pd.Timestamp(time).strftime("%Y-%m-%d"))
#     plt.savefig(f'figs/precip/plot_{time}.png')
#     plt.close()
#
# for time in spc.time.values[:100]:
#     p = spc.sel(time=time).plot.contourf(levels=30,col='encoded_dims',col_wrap=2, vmin=-10, vmax=10, cmap='RdBu',
#                                        transform=ccrs.PlateCarree(),aspect=spc.lon.shape[0] / spc.lat.shape[0],
#                                        subplot_kws={'projection': ccrs.PlateCarree()})
#
#     for i, ax in enumerate(p.axes.flat):
#         ax.coastlines()
#         ax.add_feature(cfeature.BORDERS)
#         ax.add_feature(states_provinces, edgecolor='black')
#         ax.set_extent([spc.lon.min().values, spc.lon.max().values, spc.lat.min().values, spc.lat.max().values])
#     plt.suptitle(pd.Timestamp(time).strftime("%Y-%m-%d"))
#     plt.savefig(f'figs/encoded_dims/plot_{time}.png')
#     plt.close()


spc = spc.isel(encoded_dims=slice(None, -   1))  # lose low freq which is residual
spc = spc.sel(lon=slice(-55, None), lat=slice(-10, -30))
periods = hilbert_spectrum(spc, out_type='period', apply_gauss=False)
#periods = periods.where(periods > 0, 0)
p_anomaly = xr.apply_ufunc(lambda x, y: x - y, periods.groupby('time.month'), periods.groupby('time.month').mean('time'))
p_anomaly = periods # anomaly doesnt make much sense because the low freqs capture seasonality alone
p = p_anomaly.where(p_anomaly>0, 0).groupby('time.month').mean('time').plot.contourf(col='Period', row='month',
                                                          transform=ccrs.PlateCarree(), vmin=0, vmax=3,
                                                          aspect=periods.lon.shape[0] /
                                                                 periods.lat.shape[0], cmap='Blues',
                                                          levels=50, subplot_kws={
        'projection': ccrs.PlateCarree()})
for i, ax in enumerate(p.axes.flat):

    ax.coastlines()
    ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'figs/panel_positive.png'); plt.close()

p = p_anomaly.where(p_anomaly<0, 0).groupby('time.month').mean('time').plot.contourf(col='Period', row='month',
                                                          transform=ccrs.PlateCarree(), vmin=-3, vmax=0,
                                                          aspect=periods.lon.shape[0] /
                                                                 periods.lat.shape[0], cmap='Reds_r',
                                                          levels=50, subplot_kws={
        'projection': ccrs.PlateCarree()})
for i, ax in enumerate(p.axes.flat):

    ax.coastlines()
    ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'figs/panel_negative.png'); plt.close()

p = (p_anomaly.groupby('time.season').var('time')**0.5).plot.contourf(col='Period', row='season',
                                                          transform=ccrs.PlateCarree(),vmax=18,
                                                          aspect=periods.lon.shape[0] /
                                                                 periods.lat.shape[0], cmap='nipy_spectral',
                                                          levels=50, subplot_kws={
        'projection': ccrs.PlateCarree()})
for i, ax in enumerate(p.axes.flat):

    ax.coastlines()
    ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'figs/panel_stdev.png'); plt.close()

plot_periods(periods)
print('a')
periods.groupby('time.season').mean('time').sel(Period=4, method='nearest').plot.contourf(levels=20, col='season'); plt.show()
# (periods.where(periods>0,np.nan)).argmax('Period').isel(time=0).plot(); plt.show()
# periods.Period
# periods.sel(Period=8, method='nearest').groupby('time.month').sum('time').sel(month=1).plot.contourf(
#     levels=20, cmap='RdBu')
# plt.show()
#
# periods.where(periods > 0).sel(Period=4, method='nearest').sum('time').plot();
# plt.show()
# # print('a')
# #
# # periods.argmax('Period').isel(time=10).plot()
# # plt.show()
# periods.sel(**sp, method='nearest').groupby('time.dayofyear').sum('time').plot.contourf(vmin=-0.12,
#                                                                                         vmax=0.12, levels=60,
#                                                                                         cmap='RdBu')
# plt.show()
# periods.sel(**sp, method='nearest').sum('Period').plot()
# plt.show()
# spc.sel(**sp, method='nearest').sum('encoded_dims').plot()
# plt.show()
# print('a')
# emd_grups = list(emd.groupby('encoded_dims'))
# freqs = []
# for label, group in emd_groups:
#     analytic_signal = hilbert(group.values)
#     phase = np.unwrap(np.angle(analytic_signal))
#     freq = 4 * np.diff(phase)/(2*np.pi)
#     freqs.append(freq)
# freqs = xr.DataArray(freqs, dims=['encoded_dims', 'time'],
#                    coords=dict(encoded_dims=emd.encoded_dims.values, time=emd.time.values[1:]))
#
# freqs.plot(vmin=0, vmax=1, cmap='nipy_spectral')
# plt.show()
#
# periods_array = np.linspace(50, 0.5, 500)
# k = 0
# periods = []
#
# while k < (periods_array.shape[0] - 1):
#     mask = (freqs > 1/periods_array[k]) & (freqs < 1/periods_array[k+1])
#     periods.append(emd.where(mask).sum('encoded_dims'))
#     k += 1
#     print(k)
#
# from scipy.ndimage import gaussian_filter
#
# array = xr.concat(periods, dim=pd.Index(periods_array[:-1], name='Period'))
# array.where(array > 0).plot( cmap='Blues')
# plt.show()
# array = xr.apply_ufunc(lambda x: gaussian_filter(x, 5), array)
# array.name = 'Power (mm/6h)'
# array.where(array > 0).plot.contourf(levels=100, cmap='Blues', vmax=0.12)
# plt.show()
#
# tcwv.plot()
# plt.show()
# emd.plot.line(x='time')
# plt.show()
# emd.sum('encoded_dims').plot()
# tcwv.plot()
# plt.show()
#     # frm matplotlib import cm
# # from mpl_toolkits.mplot3d import axes3d
# # spc = spc.sel(period=slice(10, 0.5))
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # x, y = np.meshgrid(spc.longitude.values, spc.period.values)
# # z=spc.values
# # ax.plot_surface(X=x, Y=y, Z=spc.values, rstride=8, cstride=8, alpha=0.3)
# # cset = ax.contour(x, y, z, zdir='z', offset=0, cmap=cm.coolwarm)
# # cset = ax.contour(x, y, z, zdir='x', offset=-200, cmap=cm.coolwarm)
# # cset = ax.contour(x, y, z, zdir='y', offset=40, cmap=cm.coolwarm)
# # plt.show()
