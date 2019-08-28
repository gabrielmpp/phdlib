import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
from scipy.ndimage.filters import gaussian_filter
filepath_re = 'data/SL_repelling_momentum.nc'
filepath_at = 'data/SL_attracting_momentum.nc'
array_re = xr.open_dataarray(filepath_re)
array_at = xr.open_dataarray(filepath_at)

array_at1 = xr.apply_ufunc(lambda x: np.log(x), (array_re ** -1) ** 0.5)
array_at2 = array_at1 - xr.apply_ufunc(lambda x: np.log(x), (array_at) ** 0.5)
new_lon = np.linspace(array_at2.longitude[0].values, array_at2.longitude[-1].values, int(array_at2.longitude.values.shape[0] * 0.4))
new_lat = np.linspace(array_at2.latitude[0].values, array_at2.latitude[-1].values, int(array_at2.longitude.values.shape[0] * 0.4))
array_at2 = array_at2.interp(latitude=new_lat, longitude=new_lon)
array_at2 = array_at2.interp(latitude=array_at1.latitude, longitude=array_at1.longitude)

# array_at1 = array_at
# array_at2 = array_re**-1
# array.isel(time=4).plot.contourf(cmap='RdBu', levels=100, vmin=0)
for time in array_at1.time.values:
    f, axarr = plt.subplots(1, 2, figsize=(30, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    array_at1.sel(time=time).plot.contourf(vmin=-8,vmax=0,levels=100, cmap='nipy_spectral', transform=ccrs.PlateCarree(),
                                           ax=axarr[0])
    axarr[0].coastlines()
    array_at2.sel(time=time).plot.contourf(vmin=-8,vmax=0,levels=100,cmap='nipy_spectral', transform=ccrs.PlateCarree(),
                                           ax=axarr[1])
    axarr[1].coastlines()
    # axarr.add_feature(states_provinces, edgecolor='gray')
    plt.savefig(f'./tempfigs/SL{time}.png')
    plt.close()
