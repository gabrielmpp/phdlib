import os  # Must be implemented before dask

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import dask
from multiprocessing.pool import ThreadPool

dask.config.set(scheduler='threads')  # Global config
dask.config.set(pool=ThreadPool(20))  # To avoid omp error
import xarray as xr
import scipy.signal as signal
import xarray as xr
import matplotlib as mpl
# mpl.use('Agg')
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


def coarsen_global_data(da: xr.DataArray, coarsen_by: float) -> xr.DataArray:
    Nlats = da.lat.values.shape[0] / coarsen_by
    Nlons = da.lon.values.shape[0] / coarsen_by
    lats = np.linspace(da.lat.values.min(), da.lat.values.max(), int(Nlats))
    lons = np.linspace(da.lon.values.min(), da.lon.values.max(), int(Nlons))
    da = da.sortby('lat')
    da = da.sortby('lon')
    da = da.interp(lat=lats, lon=lons, method='linear')
    return da


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
                           emd_array, dask='allowed'
                           )
    freq = 4 * phase.diff('time') / (2 * np.pi)
    freq = freq.unstack().load()

    if out_type == 'freq':
        return freq
    elif out_type == 'period':
        # periods_array = np.linspace(20, 4, 30)
        # periods_array = np.array([30, 25, 20, 15, 10, 5, 1])
        periods_array = np.array([90, 30, 15, 8, 4, 2, 0.5])
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
    periods_list = [90, 30, 15, 8, 4, 2]

    for time in periods.time.values:
        p = periods.sel(Period=periods_list, method='nearest').sel(time=time).plot.contourf(cmap='RdBu', vmax=10,
                                                                                            vmin=-10,
                                                                                            transform=ccrs.PlateCarree(),
                                                                                            col='Period', col_wrap=3,
                                                                                            aspect=periods.lon.shape[
                                                                                                       0] /
                                                                                                   periods.lat.shape[0],
                                                                                            levels=50, subplot_kws={
                'projection': ccrs.PlateCarree()})
        for i, ax in enumerate(p.axes.flat):
            fraction_of_power = np.abs(periods.sel(Period=periods_list[i],
                                                   method='nearest').sel(time=time)).sum(['lat', 'lon']).values / \
                                np.abs(periods.sel(time=time)).sum(['Period', 'lat', 'lon']).values
            ax.text(-44, -26, str(round(fraction_of_power * 100)) + '%')
            ax.coastlines()
            ax.add_feature(states_provinces, edgecolor='gray')
        plt.suptitle(pd.Timestamp(time).strftime("%Y-%m-%d"))
        plt.savefig(f'figs/plot_{time}.png')
        plt.close()

sp = dict(lat=-23, lon=-45, method='nearest')

# spc = xr.open_dataarray('/home/gab/phd/data/ds_emd.nc')


# ---- Line plots for SP ----#
# spc = xr.open_dataarray('/home/gab/phd/data/ds_9emd.nc').sel(**sp)
# periods = xr.open_dataarray('/home/gab/phd/data/ds_periods_9emd.nc').sel(**sp)
# chuva_reconstruida = periods.sel(time=slice('2009','2015' )).sum('Period')
# chuva_total = spc.sel(time=slice('2009', '2015')).sum('encoded_dims')
# chuva_reconstruida = chuva_reconstruida.where(chuva_reconstruida > 0, 0)
# plt.style.use('seaborn')
# plt.scatter(chuva_reconstruida.values, chuva_total.values, alpha=0.3)
# plt.plot([0,110], [0,110], color='black')
# plt.xlabel('Reconstructed rainfall (mm/day)')
# plt.ylabel('Observed rainfall (mm/day)')
# plt.show()
# chuva_total.plot()
# chuva_reconstruida.plot()
# plt.ylabel('Rainfall (mm/day)')
# plt.legend(['Observed rainfall', 'Reconstructed rainfall'])
# plt.show()
# spc.plot(row='encoded_dims', aspect=5)
# plt.show()
# periods.plot(row='Period', aspect=3)
# ---- computing Hilbert ----- #
#spc = xr.open_dataarray('/home/users/gmpp/ds_emd.nc')
# #
# spc = spc.isel(encoded_dims=slice(None, -   1))  # lose low freq which is residual
# periods = hilbert_spectrum(spc, out_type='period', apply_gauss=False)
# periods.to_netcdf('/home/users/gmpp/ds_periods.nc')

# periods = xr.open_dataarray('/home/users/gmpp/ds_periods.nc')
periods = xr.open_dataarray('/home/gab/phd/data/ds_periods_9emd.nc')
spc = xr.open_dataarray('/home/gab/phd/data/ds_9emd.nc')

periods = periods.chunk({'Period': 6})

stdev = (periods.where(periods>0).groupby('time.season').mean('time')) #var('time')**0.5)
# p = stdev.plot.contourf(col='Period',
#                         row='season', transform=ccrs.PlateCarree(),vmax=15,
#                         aspect=periods.lon.shape[0] / periods.lat.shape[0], cmap='nipy_spectral',
#                         levels=50, subplot_kws={'projection': ccrs.PlateCarree()})
# for i, ax in enumerate(p.axes.flat):
#
#     ax.coastlines()
#     ax.add_feature(states_provinces, edgecolor='gray')
#
# plt.savefig(f'/home/users/gmpp/panel_stdev.png'); plt.close()

stdev_prec = spc.sum('encoded_dims').groupby('time.season').mean('time')#var('time')**0.5
spc = None
periods = None
period1 = stdev.sel(Period=2).drop('Period')
period2 = stdev.sel(Period=4).drop('Period') + stdev.sel(Period=8).drop('Period')
period3 = stdev.sel(Period=15).drop('Period') + stdev.sel(Period=30).drop('Period') + stdev.sel(Period=90).drop('Period')
stdev_ = xr.concat([stdev_prec, period1, period2, period3], dim=pd.Index(['total', '2', '2 - 8', '8 - 90'], name='Period'))
# stdev_ = coarsen_global_data(stdev_, coarsen_by=2.)
stdev_ = stdev_.where(stdev_>0, np.nan)

p = stdev_.plot.contourf(col='Period',levels=np.arange(3, 12, 1),
                        row='season', transform=ccrs.PlateCarree(),vmax=12, vmin=3,
                        aspect=stdev_.lon.shape[0] / stdev_.lat.shape[0], cmap='nipy_spectral_r',
                         subplot_kws={'projection': ccrs.PlateCarree()},
                        cbar_kwargs=dict(orientation="right"))
for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
plt.show()
#     ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'/home/users/gmpp/panel_stdev2.pdf'); plt.close()


import seaborn as sns
levels = np.arange(3, 12, 0.5)
palette = sns.color_palette("nipy_spectral_r", levels.shape[0])
p = stdev_.sel(season='DJF').plot(col='Period', cmap='nipy_spectral_r', vmin=3, vmax=12,
                        col_wrap=2, transform=ccrs.PlateCarree(),
                        aspect=stdev_.lon.shape[0] / stdev_.lat.shape[0],
                         subplot_kws={'projection': ccrs.PlateCarree()})

for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
plt.savefig('figs/panel_period.pdf')
plt.close()
# stdev_ = coarsen_global_data(stdev_, coarsen_by=3)
p = stdev_.sel(season='DJF').plot.contour(col='Period',
                        col_wrap=2, transform=ccrs.PlateCarree(),levels=[6, 6.1],
                        aspect=stdev_.lon.shape[0] / stdev_.lat.shape[0],
                         subplot_kws={'projection': ccrs.PlateCarree()})

for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
plt.show()
#     ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'/home/users/gmpp/panel_stdev3.pdf'); plt.close()