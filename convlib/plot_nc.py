import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

filepath = '/home/users/gmpp/SL.nc'

array = xr.open_dataarray(filepath)

for time in array.time.values:
    f, axarr = plt.subplots(1, 1, figsize=(14, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    array.sel(time=time).plot.contourf(vmin=0,levels=40,vmax=1,cmap = 'nipy_spectral',transform=ccrs.PlateCarree(), ax=axarr[0])
    axarr[0].coastlines()
    axarr[0].add_feature(states_provinces, edgecolor='gray')
    plt.savefig(f'./tempfigs/SL{time}.png')
    plt.close()