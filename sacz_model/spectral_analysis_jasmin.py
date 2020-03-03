import scipy.signal as signal
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from phdlib.utils.xrumap import autoencoder
import numpy as np
from pathlib import Path

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
    try:
        array = array.sel({latname: slice(lat1, lat2), lonname: slice(lon1, lon2)})
    except:
            lonmask = (array[lonname] < lon2) & (array[lonname] > lon1)
            latmask = (array[latname] < lat2) & (array[latname] > lat1)
            array = array.where(lonmask, drop=True)
            array = array.where(latmask, drop=True)
    return array
era5path = Path('/gws/nopw/j04/primavera1/observations/ERA5/')
ftlepath = Path('/group_workspaces/jasmin4/upscale/gmpp/convzones/')
years = range(2000, 2001)
tcwvs = list()
for k, year in enumerate(years):
    print(f"Reading year {year}")

    tcwv_temp = xr.open_dataarray(
        era5path     / 'tcwv_ERA5_6hr_{yr}010100-{yr}123118.nc'.format(yr=str(year)))
    tcwv_temp = tcwv_temp.assign_coords(longitude=(tcwv_temp.coords['longitude'].values + 180) % 360 - 180)
    print('sorting')
    #tcwv_temp = tcwv_temp.sortby('longitude').sortby('latitude')
    print(tcwv_temp)
    print('slicing')
    tcwv_temp = tcwv_temp.sel(latitude=slice(-15, -25))
    print(tcwv_temp)

    # tcwv_temp['month'] = 'time', [pd.to_datetime(x).strftime('%m') for x in tcwv_temp.time.values]
    tcwv_temp = tcwv_temp.mean('latitude')


    tcwvs.append(tcwv_temp)
tcwv = xr.concat(tcwvs, dim='time')
enc = autoencoder(dims_to_reduce=None, alongwith=['time'], mode='emd')  # TODO WARNING TEST MODE
enc = enc.fit(tcwv)
tcwv_enc = enc.transform(tcwv, mode='emd')
tcwv_enc.to_netcdf('tcwv_emd.nc')
#
# tcwv = tcwv.sortby('longitude')
# longroups = tcwv.groupby('longitude')
# freqlist = []
# k = 0
# for label, group in list(longroups):
#     print(k)
#     freq, power = signal.periodogram(group.values, axis=0)
#     temp_array = xr.DataArray(power, dims=['freq'], coords=dict(freq=freq))
#     freqlist.append(temp_array)
#     k += 1
#
# freqarray = xr.concat(freqlist, dim=tcwv.longitude)
#
# freqarray = freqarray.isel(freq=slice(1, None)).assign_coords(freq=0.25*(1/freqarray.freq.values)[1:]).rename(freq='period')
# freqarray = xr.apply_ufunc(lambda x: np.log(x), freqarray)
# freqarray = freqarray.transpose()
# #freqarray = freqarray.groupby('time.month').sum('time')
# freqarray.to_netcdf('freqarray.nc')
