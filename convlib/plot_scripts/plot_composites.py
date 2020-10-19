import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
import matplotlib
import cmasher as cmr
import matplotlib
import numpy as np
matplotlib.rcParams['hatch.color'] = 'gray'

data_dir = '/home/gab/phd/data/composites_cz/'
MAG = xr.open_dataset('~/phd/scripts/phdlib/convlib/data/xarray_mair_grid_basins.nc')
MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})


daa = xr.open_dataset(data_dir + 'basin_composites.nc')
ds_seasonal_avgs = xr.open_dataset(data_dir + 'ds_seasonal_avgs.nc')
mask_tiete = MAG['Tiete'].interp(latitude=daa.latitude, longitude=daa.longitude, method='nearest')
mask_uruguai = MAG['Uruguai'].interp(latitude=daa.latitude, longitude=daa.longitude, method='nearest')
mask_tiete = mask_tiete.sel(latitude=slice(-30, -15), longitude=slice(-55, -30))
mask_uruguai = mask_uruguai.sel(latitude=slice(-35, -15), longitude=slice(-65, -30))
masks = [mask_tiete, mask_uruguai]
basins = ['Tiete', 'Uruguai']
seasons = ['DJF', 'JJA']

daa_days_of_cz = xr.open_dataarray(data_dir + 'basin_composites_days_of_cz.nc')

# ---- Avgs

t_threshold = 2.807    # 99% bicaudal test
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},
                                          gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
k=0
l=0
titles='ab'
for season in seasons:
    ax = axs.flatten()[k]
    u = ds_seasonal_avgs['u_season'].sel(season=season)
    v = ds_seasonal_avgs['v_season'].sel(season=season)
    u = u.dropna('longitude', how='all')
    u = u.dropna('latitude', how='all')
    v = v.dropna('longitude', how='all')
    v = v.dropna('latitude', how='all')
    u = u.coarsen(latitude=10, longitude=10, boundary='trim').mean()
    v = v.coarsen(latitude=10, longitude=10, boundary='trim').mean()
    magnitude = np.sqrt(u ** 2 + v ** 2).values
    da_to_plot = ds_seasonal_avgs['pr_season'].sel(season=season)
    da_to_plot = da_to_plot.dropna('longitude', how='all')
    da_to_plot = da_to_plot.dropna('latitude', how='all')
    p = (da_to_plot*86400/4).plot(ax=ax, cmap=cmr.freeze_r, transform=ccrs.PlateCarree(), vmax=12, vmin=0,
                                    add_colorbar=False)
    s = ax.quiver(u.longitude.values, u.latitude.values, u.values, v.values,
                      linewidth=.8, color='red',
                      transform=ccrs.PlateCarree(), scale=4000)
    qk = ax.quiverkey(s, 0.65, 0.75, 200, r'$200 \frac{kg}{ms}$', labelpos='E',
                       coordinates='figure')
    ax.set_title(None)
    # s.lines.set_clim(0, 200)
    # s.arrows.set_clim(0, 200)

    ax.set_xlim(da_to_plot.longitude.min().values, da_to_plot.longitude.max().values)
    ax.set_ylim(da_to_plot.latitude.min().values, da_to_plot.latitude.max().values)
    ax.set_title(titles[k], loc='left')
    ax.coastlines(color='black')
    k+=1
    l+=1

cbar = fig.colorbar(p, ax=axs, shrink=0.6)
# cbar2 = fig.colorbar(s.lines, ax=axs, shrink=0.7, orientation='horizontal')
gl = axs[0].gridlines(draw_labels=True)

gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.ylabels_right = False
gl.xlabels_top = False
gl.xlabels_bottom = True
gl.xlines = False
gl.ylines = False
gl.xlocator = ticker.FixedLocator([-80, -60, -40, -20])

gl = axs[1].gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_left = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-80, -60, -40, -20])
gl.xlines = False
gl.ylines = False

cbar.ax.set_ylabel(r'Rainfall $(\frac{mm}{day})$')
# cbar2.ax.set_xlabel('VIMF' + r' $(\frac{kg}{m s})$')
plt.savefig('../tempfigs/avg_precip_wind.png', dpi=600,
            transparent=True, pad_inches=.2,  bbox_inches='tight'
            )

plt.close()


# ---- Precip anomalies

t_threshold =  1.960    # 99% bicaudal test
fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()},
                                          gridspec_kw={'hspace': 0.2, 'wspace': 0.})
t_threshold_wind = 1.960
k = 0
l = 0
titles = 'abcd'
for basin in basins:
    for season in seasons:
        n = daa_days_of_cz.sel(basin=basin, season=season).values

        ax = axs.flatten()[k]
        u = daa['u'].sel(basin=basin, season=season)
        v = daa['v'].sel(basin=basin, season=season)
        t_test_u = (n ** 0.5) * \
                 u / ds_seasonal_avgs['u_var_season'].sel(season=season) ** 0.5
        t_test_v = (n ** 0.5) * \
                 v / ds_seasonal_avgs['v_var_season'].sel(season=season) ** 0.5
        t_test_u.where(t_test_u)
        t_test_uv = t_test_u.where(np.abs(t_test_u) > t_threshold_wind)
        t_test_uv = t_test_uv.where(np.abs(t_test_v) > t_threshold_wind)
        t_test_wind = ~xr.ufuncs.isnan(t_test_uv)
        u = u.where(t_test_wind)
        v = v.where(t_test_wind)
        u = u.dropna('longitude', how='all')
        u = u.dropna('latitude', how='all')
        v = v.dropna('longitude', how='all')
        v = v.dropna('latitude', how='all')
        u = u.coarsen(latitude=10, longitude=10, boundary='trim').mean()
        v = v.coarsen(latitude=10, longitude=10, boundary='trim').mean()

        magnitude = np.sqrt(u ** 2 + v ** 2).values
        da_to_plot = daa['pr'].sel(basin=basin, season=season)
        da_to_plot = da_to_plot.dropna('longitude', how='all')
        da_to_plot = da_to_plot.dropna('latitude', how='all')
        t_test = (n ** 0.5) *\
                 da_to_plot / ds_seasonal_avgs['pr_var_season'].sel(season=season) ** 0.5
        p = (da_to_plot*86400).plot(ax=ax, cmap=cmr.fusion, transform=ccrs.PlateCarree(), vmax=12, vmin=-12,
                                    add_colorbar=False)
        np.abs(t_test).plot.contourf(ax=ax, hatches=['  ', '......'], linewidths=.8,
                             levels=[0, t_threshold], alpha=0, add_colorbar=False)
        s = ax.quiver(u.longitude.values, u.latitude.values, u.values, v.values,
                      color='blue', edgecolor='red', headwidth=4, linewidth=.15,
                      transform=ccrs.PlateCarree(), scale=800)
        qk = ax.quiverkey(s, 0.65, 0.92, 50, r'$50 \frac{kg}{ms}$', labelpos='E',
                          coordinates='figure')

        masks[l].plot.contour(levels=[0, 0.5], cmap='red', ax=ax, transform=ccrs.PlateCarree(),
                                add_colorbar=False, linewidths=.6)
        ax.set_xlim(da_to_plot.longitude.min().values, da_to_plot.longitude.max().values)
        ax.set_title(titles[k], loc='left')
        ax.coastlines(color='black')
        k+=1
    l+=1

cbar = fig.colorbar(p, ax=axs, shrink=.8)
gl = axs[0, 0].gridlines(draw_labels=True)

gl.yformatter = LATITUDE_FORMATTER
gl.ylabels_right = False
gl.xlabels_top = False
gl.xlabels_bottom=False
gl.xlines = False
gl.ylines = False

gl = axs[1,1].gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_left = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-80, -60, -40, -20])
gl.xlines = False
gl.ylines = False

cbar.ax.set_ylabel(r'Rainfall anomaly $(\frac{mm}{day})$')
plt.savefig('../tempfigs/Anomalies_precip_basin.png', dpi=600,
            transparent=True, bbox_inches='tight', pad_inches=.2)

plt.close()

# ---- Geop anomalies

fig, axs = plt.subplots(4, 2, subplot_kw={'projection': ccrs.PlateCarree()},
                        gridspec_kw={'hspace': .3, 'wspace': -.4})
k=0
l=0
t = 0
titles=['a) Summer, lag 0',
        'c) Summer, lag -3 days',
        'b) Winter, lag 0 ',
        'd) Winter, lag -3 days',
        'e) Summer, lag 0',
        'g) Summer, lag -3 days',
        'f) Winter, lag 0 ',
        'h) Winter, lag -3 days',
        ]

font = {'size': 6}

matplotlib.rc('font', **font)
for basin in basins:
    k=0
    for season in seasons:
        ax = axs[l, k]
        da_to_plot = daa['z'].sel(basin=basin, season=season)
        n = daa_days_of_cz.sel(basin=basin, season=season).values
        t_test = ( n ** 0.5) * \
                 da_to_plot / ds_seasonal_avgs['z_var_season'].sel(season=season) ** 0.5
        p = (da_to_plot/10).plot(ax=ax, cmap=cmr.fusion, transform=ccrs.PlateCarree(),
                                    add_colorbar=False)
        np.abs(t_test).plot.contourf(ax=ax, hatches=[' ', '...'], cmap='white', linewidths=.8,
                             levels=[0, t_threshold], alpha=0, add_colorbar=False)
        masks[int(l/2)].plot.contour(levels=[0, 0.5], cmap='red', ax=ax, transform=ccrs.PlateCarree(),
                                add_colorbar=False, linewidths=.8)
        ax.set_title(titles[t], loc='left')
        ax.coastlines()
        t+=1
        ax = axs[l+1, k]
        da_to_plot = daa['z3'].sel(basin=basin, season=season)
        n = daa_days_of_cz.sel(basin=basin, season=season).values
        t_test = ( n ** 0.5) * \
                 da_to_plot / ds_seasonal_avgs['z_var_season'].sel(season=season) ** 0.5
        p = (da_to_plot/10).plot(ax=ax, cmap=cmr.fusion, transform=ccrs.PlateCarree(),
                                    add_colorbar=False)
        np.abs(t_test).plot.contourf(ax=ax, hatches=[' ', '...'], cmap='white', linewidths=.8,
                             levels=[0, t_threshold], alpha=0, add_colorbar=False)

        masks[int(l/2)].plot.contour(levels=[0, 0.5], cmap='red', ax=ax, transform=ccrs.PlateCarree(),
                                add_colorbar=False, linewidths=.8)
        ax.set_title(titles[t], loc='left')
        ax.coastlines()
        t+=1
        k+=1
    l+=2


cbar = fig.colorbar(p, ax=axs, shrink=0.7)

gl = axs[0, 0].gridlines(draw_labels=True)

gl.yformatter = LATITUDE_FORMATTER
gl.ylabels_right=False
gl.xlabels_top=False
gl.xlabels_bottom=False
gl.xlines = False
gl.ylines = False
gl.ylocator = ticker.FixedLocator([-60, -40, -20, 0, 20])

gl = axs[3, 1].gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_left = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-120, -80,   -40,])
gl.xlines = False
gl.ylines = False
cbar.ax.set_ylabel('Geopotential height anomaly (m)')
plt.savefig('../tempfigs/Anomalies_geop_basin.png', dpi=600,
            transparent=True, bbox_inches='tight', pad_inches=0.1)

plt.close()


