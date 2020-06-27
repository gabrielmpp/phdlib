import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_cfc512b7-f65b-480b-9935-91349b9f2e75/'
file = 'SL_attracting_lcstimelen_8_partial_0.nc'
time = '2000-01-03T18:00:00'
da = xr.open_dataarray(path + file)
# da = da.resample(time='1D').mean()
da = da.sel(time=time)
da = np.log(np.sqrt(da)) / 2 #days


fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Geostationary(
    central_longitude=-75)), figsize=[14, 14])
da.plot(cmap='inferno', ax=axs, transform=ccrs.PlateCarree(), vmin=0, vmax=2)
axs.set_global()
axs.coastlines(color='b')
plt.savefig('tempfigs/snapshot.png')
plt.close()