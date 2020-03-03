import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
import cartopy.crs as ccrs

def latlonsel(array, lat, lon, latname='lat', lonname='lon'):
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


    if isinstance(lat, slice):
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
    array = array.where(lonmask, drop=True).where(latmask, drop=True)
    return array


infolder_path = Path('~/phd/data/FTLE_ERA5/')
era5path = Path('/gws/nopw/j04/primavera1/observations/ERA5/')
ftlepath = Path('/group_workspaces/jasmin4/upscale/gmpp/convzones/')
years = range(1981, 2001)
month = '01'
ftles = list()
precips = list()
for k, year in enumerate(years):
    print(f"Reading year {year}")

    ftles.append(
        xr.open_dataarray(
            ftlepath / 'SL_repelling_{}_lcstimelen_16_v2.nc'.format(str(year))
        ).sel(latitude=slice(None, 10), time='{yr}-{mn}'.format(yr=str(year), mn=month))
    )
    precip_temp = xr.open_dataarray(
        era5path     / 'pr_ERA5_6hr_{yr}010100-{yr}123118.nc'.format(yr=str(year)))
    precip_temp = precip_temp.assign_coords(longitude=(precip_temp.coords['longitude'].values + 180) % 360 - 180)
    precip_temp = precip_temp.sel(time=ftles[k].time, method='nearest')
    precip_temp = latlonsel(precip_temp, lat=slice(ftles[k].latitude.min().values, ftles[k].latitude.max().values),
                       lon=slice(ftles[k].longitude.min().values, ftles[k].longitude.max().values), latname='latitude',
                       lonname='longitude')
    precip_temp = precip_temp.sel(latitude=ftles[k].latitude, longitude=ftles[k].longitude, method='nearest')
    precip_temp = precip_temp.assign_coords(latitude=ftles[k].latitude, longitude=ftles[k].longitude)
    precips.append(precip_temp)

print("start concat")
precip = xr.concat(precips, dim='time')
print("done precip")
ftle = xr.concat(ftles, dim='time')
print("done ftle")
print("done concat")

threshold = ftle.quantile(0.9)
freq = ftle.where(ftle > threshold, 0)
freq = freq.where(ftle <= threshold, 1)
freq = freq.mean('time')
print(freq)

qs = precip.where(ftle > threshold, drop=True).quantile(0.12, dim='time')
fig = plt.figure(figsize=[12, 10])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
qs.plot(vmax=0.00005, cmap="viridis",vmin=0, ax=ax, add_colorbar=False)
FRQ = freq.plot.contour(cmap='Reds', ax=ax, levels=np.arange(0, 1, 0.12))
ax.clabel(FRQ, inline=1, fontsize=10)
ax.coastlines()
plt.savefig(f'/home/users/gmpp/quantiles_cz_month_{month}.png')
plt.close()

fig = plt.figure(figsize=[12, 10])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
precip.quantile(0.1, dim='time').plot(ax=ax, vmin=0,add_colorbar=False, vmax=0.0001)
FRQ = freq.plot.contour(cmap='Reds', ax=ax, levels=np.arange(0, 1, 0.12))
ax.clabel(FRQ, inline=1, fontsize=10)
ax.coastlines()
plt.savefig(f'/home/users/gmpp/quantiles_precip_{month}.png')
plt.close()

# untested!
# untested!
def covariance(x, y, dim=None):
    valid_values = x.notnull() & y.notnull()
    valid_count = valid_values.sum(dim)

    demeaned_x = (x - x.mean(dim)).fillna(0)
    demeaned_y = (y - y.mean(dim)).fillna(0)

    return xr.dot(demeaned_x, demeaned_y, dims=dim) / valid_count


def correlation(x, y, dim=None):
    # dim should default to the intersection of x.dims and y.dims
    return covariance(x, y, dim) / (x.std(dim) * y.std(dim))


corr = correlation(ftle, precip, 'time')
fig = plt.figure(figsize=[12, 10])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
corr.plot(ax=ax, cmap='viridis', vmin=0, vmax=0.2)
ax.coastlines()
plt.savefig(f'/home/users/gmpp/correlation_ftle_precip_{month}.png')
plt.close()
