import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from mpl_toolkits.axes_grid1 import make_axes_locatable
path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_cfc512b7-f65b-480b-9935-91349b9f2e75/'
file = 'SL_attracting_lcstimelen_8_partial_584.nc'
tcwv_path = '/gws/nopw/j04/primavera1/observations/ERA5/tcwv_ERA5_6hr_2006010100-2006123118.nc'
cpc_path = '~/phd_data/precip_1979a2017_CPC_AS.nc'
time = '2006-01-03'

tcwv = xr.open_dataarray(tcwv_path)
tcwv = tcwv.sel(time=time)
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
cpc = xr.open_dataarray(cpc_path)
cpc = cpc.rename({'lon': 'longitude', 'lat': 'latitude'})
cpc = cpc.assign_coords(longitude=(cpc.coords['longitude'].values + 180) % 360 - 180)

da = xr.open_dataarray(path + file)
da = da.sel(time='2006-01-03')
da_pr = cpc.sel(latitude=da.latitude, longitude=da.longitude, method='nearest').\
    sel(time='2006-01-03')
tcwv = tcwv.sel(latitude=da.latitude, longitude=da.longitude, method='nearest').mean('time')
da = da.isel(time=-3)

da = np.log(np.sqrt(da)) / 2 #days
tcwv.name='Total column water vapour (kg/mg²)'
da.name='FTLE (1/day)'
fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection=ccrs.PlateCarree(
    central_longitude=-75)), figsize=[10, 5])
c1 = da.plot.contourf(cmap=cmr.rainforest, ax=axs[0], levels=31,transform=ccrs.PlateCarree(), vmin=0.5, vmax=2,
                      add_colorbar=True, cbar_kwargs={'format': '%1.1f', 'shrink': 0.8})
c2 = tcwv.plot.contourf(cmap=cmr.gem, ax=axs[1], levels=31,transform=ccrs.PlateCarree(), vmin=5, vmax=60,
                        add_colorbar=True, cbar_kwargs={'format': '%1.1f', 'shrink': 0.8})
c3 = da_pr.plot.contour( ax=axs[1], levels=[10, ], colors='black', transform=ccrs.PlateCarree())
axs[1].clabel(c3, inline=1, fontsize=10, fmt='%1.0f')
for cl in c1.collections: cl.set_edgecolor('face')
for cl in c2.collections: cl.set_edgecolor('face')
# axs.set_global()
axs[0].set_title(None)
axs[1].set_title(None)
axs[0].coastlines(color='red')
axs[1].coastlines(color='red')
plt.subplots_adjust(wspace=0.1)
fig.tight_layout()
plt.savefig('tempfigs/snapshot.pdf')
plt.close()