import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
dir = '/home/gab/phd/data/corrs_ftle/'
corr_rain = dir + 'corr_rainfall.nc'
corr_rain3 = dir + 'corr_rain_3.nc'
corr_tcwv = dir + 'corr_tcwv.nc'
corr_tcwv3 = dir + 'corr_tcwv_3.nc'

da_rain = xr.open_dataarray(corr_rain)
da_tcwv = xr.open_dataarray(corr_tcwv)
da_rain3 = xr.open_dataarray(corr_rain3)
da_tcwv3 = xr.open_dataarray(corr_tcwv3)
n = 4 * 90 * 30
t_rain = np.abs(da_rain*np.sqrt((n-2)/(1-da_rain**2)))
t_tcwv = np.abs(da_tcwv*np.sqrt((n-2)/(1-da_tcwv**2)))
t_rain3 = np.abs(da_rain3*np.sqrt((n-2)/(1-da_rain3**2)))
t_tcwv3 = np.abs(da_tcwv3*np.sqrt((n-2)/(1-da_tcwv3**2)))

t_threshold = 2.807  # 99.% confidence two tailed


fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})
cmap = cmr.fusion

da_rain.plot(ax=axs[0, 0], transform=ccrs.PlateCarree(), cmap=cmap, vmin=-0.3, vmax=0.3,
             add_colorbar=False)
t_rain.plot.contourf(ax=axs[0, 0], hatches=['/////', ' '], levels=[0, t_threshold],cmap='gray',
                     alpha=0,add_colorbar=False)

p=da_tcwv.plot(ax=axs[0, 1], transform=ccrs.PlateCarree(), cmap=cmap, vmin=-0.3, vmax=0.3,add_colorbar=False )
t_tcwv.plot.contourf(ax=axs[0, 1], hatches=['/////', ' '],cmap='gray',
                     levels=[0, t_threshold], alpha=0,add_colorbar=False)
da_rain3.plot(ax=axs[1, 0], transform=ccrs.PlateCarree(), cmap=cmap, vmin=-0.3, vmax=0.3,
             add_colorbar=False)

t_rain3.plot.contourf(ax=axs[1, 0], hatches=['/////', ' '], levels=[0, t_threshold],cmap='gray',
                     alpha=0,add_colorbar=False)

da_tcwv3.plot(ax=axs[1, 1], transform=ccrs.PlateCarree(), cmap=cmap,
             vmin=-0.3, vmax=0.3,add_colorbar=False)
t_tcwv3.plot.contourf(ax=axs[1, 1], hatches=['/////', ' '],cmap='gray',
                     levels=[0, t_threshold], alpha=0,add_colorbar=False)

cbar_ax = plt.colorbar(p, ax=axs, location='right', shrink=0.6)
cbar_ax.ax.set_ylabel('Pearson correlation')
labels = 'abcd'
for idx, ax in enumerate(axs.flatten()):
    ax. coastlines(color='black')
    ax.set_title(labels[idx], loc='left')

gl = axs[0, 0].gridlines(draw_labels=True)

gl.yformatter = LATITUDE_FORMATTER
gl.ylabels_right=False
gl.xlabels_top=False
gl.xlabels_bottom=False
gl.xlines = False
gl.ylines = False

gl = axs[1,1].gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_left = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-80, -60, -40])
gl.xlines = False
gl.ylines = False
plt.savefig('../tempfigs/correlations_rain_tcwv.png',
            transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
plt.close()