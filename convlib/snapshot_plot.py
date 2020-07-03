import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_cfc512b7-f65b-480b-9935-91349b9f2e75/'
file = 'SL_attracting_lcstimelen_8_partial_0.nc'
sp = '/gws/nopw/j04/primavera1/observations/ERA5/viwv_ERA5_6hr_2000010100-2000123118.nc'
time = '2000-01-03T18:00:00'
da = xr.open_dataarray(path + file)
da_sp = xr.open_dataarray(sp)
# da = da.resample(time='1D').mean()
da = da.sel(time=time)
da_sp = da_sp.sel(time=time)

da = np.log(np.sqrt(da)) / 2 #days


fig, axs = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Geostationary(
    central_longitude=-75)))
c= da.plot.contourf(cmap=cmr.rainforest, ax=axs, levels=11,transform=ccrs.PlateCarree(), vmin=0, vmax=2)
for cl in c.collections: cl.set_edgecolor('face')
# axs.set_global()
axs.coastlines(color='red')
plt.savefig('tempfigs/snapshot.pdf')
plt.close()