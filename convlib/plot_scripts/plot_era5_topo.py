import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
import cartopy.feature as cfeature

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
locations = {
    'Brazil': [-52, -15],
    'Argentina': [-68, -34],
    'Uruguay': [-56, -33],
    'Paraguay': [-58, -23]
}
data_dir = '/home/gab/phd/data/ERA5/geopotential_orography.nc'

da = xr.open_dataarray(data_dir)
da = da.assign_coords(longitude=(da.coords['longitude'].values + 180) % 360 - 180)
da = da.sortby('longitude')
da = da.sortby('latitude')
da = da.sel(latitude=slice(-56, 15), longitude=slice(-90, -30))
da = da / 9.80665  # grav

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
p = da.plot(ax=ax, vmin=-200, vmax=4000, cmap='terrain', add_colorbar=False)
ax.set_title('')
cbar = fig.colorbar(p, ax=ax, shrink=0.6)
# cbar2 = fig.colorbar(s.lines, ax=axs, shrink=0.7, orientation='horizontal')
gl = ax.gridlines(draw_labels=True)

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylabels_right = False
gl.xlabels_top = False
gl.xlabels_bottom = True
gl.xlines = False
gl.ylines = False
gl.xlocator = ticker.FixedLocator([-80, -60, -40, -20])

ax.add_feature(cfeature.BORDERS, linewidth=.5)
ax.add_feature(states_provinces, edgecolor='gray', linewidth=.5)
cbar.ax.set_ylabel(r'Elevation above sea level (m)')
ax.coastlines()
for location in locations.keys():
    x=locations[location][0]
    y=locations[location][1]
    ax.text(x=x, y=y, s=location, fontsize=6, color='white', bbox={'boxstyle': 'square', 'alpha':.3, 'color':'black'},
            )
    ax.scatter(x, y, color='red', marker='.')
plt.savefig('../tempfigs/era5_topo.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )
plt.close()