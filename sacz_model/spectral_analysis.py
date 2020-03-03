import scipy.signal as signal
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from phdlib.utils.xrumap import autoencoder as xru

def latlonsel(array, array_like = None, lat = None, lon = None, latname='lat', lonname='lon'):
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
    #array = dict(array.groupby('time.season'))['DJF']
    groups = list(array.groupby('stacked'))
    power_list = []
    for label, group in groups:
        freq, power = signal.periodogram(group.values, axis=0)
        power_list.append(xr.DataArray(power, dims=['freq'], coords=dict(freq=freq)))

    array = xr.concat(power_list, dim=array.stacked)
    array = array.unstack()
    array = xr.apply_ufunc(lambda x: np.log(x), array)
    array = array.assign_coords(freq=1/array.freq).rename(dict(freq='period'))
    # array = array.coarsen(period=Z, boundary='trim').mean()
    array.sel(A_amazon=10, period=slice(15, 2)).plot.contourf(levels=30, vmin=-26, vmax=-20, cmap='plasma')
    plt.show()
    array.sel(A_amazon=7, period=slice(15, 2)).plot.contourf(levels=30, vmin=-26, vmax=-20, cmap='plasma')
    plt.show()
    (array.sel(A_amazon=10, period=slice(15, 2)) - array.sel(A_amazon=7, period=slice(15, 2))).plot.contourf(levels=30,vmin=-3.5, vmax=3.5, cmap='RdBu')
    plt.show()
    array.sel(A_amazon=10, period=4, method='nearest').plot.line(x='coupling')
    plt.show()


    # qqplot


    ds = xr.open_dataset('data/ds.nc')
    idxs = np.array([pd.to_datetime(t).month in [1, 2, 12] for t in ds.time.values])
    ds = ds.sel(time=idxs)
    array = ds.P.resample(time='1D').sum()* 86400
    arrayQ = ds.Q.resample(time='1D').sum()#* 86400
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

def hilbert_spectrum(emd_array, out_type='period'):
    from scipy.signal import hilbert

    from joblib import Parallel, delayed
    def run_hilbert(group):
        analytic_signal = hilbert(group)
        phase = np.unwrap(np.angle(analytic_signal))
        #  freq = 4 * np.diff(phase) / (2 * np.pi)
        return phase

    dask_array = emd_array.chunk(dict(encoded_dims=1, lat=1, lon=1))
    temp_array = dask_array.stack({'stacked': ['encoded_dims', 'lat', 'lon']})
    temp_array = xr.apply_ufunc(lambda x: run_hilbert(x),
                                temp_array.groupby('stacked'),
                                dask='allowed',
                                )
    temp_array =
    temp_array = temp_array.unstack().load()
    if out_type == 'freq':
        return temp_array
    elif out_type == 'period':
        periods_array = np.linspace(10, 0.5, 10)
        k = 0
        periods = []

        while k < (periods_array.shape[0] - 1): # TODO make it faster
            mask = (temp_array > 1 / periods_array[k]) & (temp_array < 1 / periods_array[k + 1])
            periods.append(temp_array.where(mask).sum('encoded_dims'))
            k += 1
            print(k)

        from scipy.ndimage import gaussian_filter

        array = xr.concat(periods, dim=pd.Index(periods_array[:-1], name='Period'))
        return array
    else:
        raise ValueError(f'out_type {out_type} not supported.')

spc = xr.open_dataarray('~/phd/data/prec_emd_for_mair.nc')
spc = spc.isel(lat=slice(None, 2), lon=slice(None, 2))
periods = hilbert_spectrum(spc)
periods = periods.chunk()
periods.isel(Period=0, time=0).plot()
plt.show()
print('a')
# emd_groups = list(emd.groupby('encoded_dims'))
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
