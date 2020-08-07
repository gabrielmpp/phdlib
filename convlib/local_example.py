import cartopy.crs as ccrs
import xarray as xr
from LagrangianCoherence.LCS import LCS
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker

from matplotlib.gridspec import GridSpec as GS
import sys


# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
pr_filepath = basepath + 'pr_ERA5_6hr_2020010100-2020123118.nc'

# timesel = sys.argv[1]
timesel = slice('2020-01-20', '2020-01-28')
subdomain = {'latitude': slice(-40, 15),
             'longitude': slice(-85, -32)}
u = xr.open_dataarray(u_filepath, chunks={'time': 140})
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')

u = u.sel(latitude=slice(-75, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
u = u.sel(time=timesel)
u = u.load()

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-75, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
v = v.sel(time=timesel)
v = v.load()

tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-75, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
tcwv = tcwv.sel(time=timesel)
tcwv = tcwv.load()

pr = xr.open_dataarray(pr_filepath, chunks={'time': 140})
pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
pr = pr.sortby('longitude')
pr = pr.sortby('latitude')
pr = pr.sel(subdomain)
pr = pr.sel(time=timesel)
pr = pr.load()
pr = pr * 1000  # Originally in metres
pr = pr #  Converting to mm/h
pr.isel(time=19).plot.contour(levels=[1, 5, 10], cmap=cmr.freeze)
plt.show()
# rain = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')

u = u/tcwv
v = v/tcwv
u.name = 'u'
v.name = 'v'
ds = xr.merge([u, v])
# ---- Running LCS ---- #
ntimes = 30*4
ftle_list = []

for dt in range(ntimes):
    timeseq = np.arange(0, 8) + dt
    # lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, return_det=True)
    # det = lcs(ds.isel(time=timeseq))
    # det = np.sqrt(det)
    # potential_rainfall = (tcwv.isel(time=3) / (-6*3600*timeseq.shape[0])) * (1 - det)/det
    # potential_rainfall = - ( 2 * tcwv / (-6*3600*5) ) * (det - 1) / (det + 1)  # [mm / s]
    # potential_rainfall = potential_rainfall * 86400  # conversion to [mm / day]

    lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, return_dpts=True,
                  gauss_sigma=None)
    ftle, x_departure, y_departure = lcs(ds.isel(time=timeseq), s=1e5, resample='3H')

    x_departure = x_departure.sel(subdomain)
    y_departure = y_departure.sel(subdomain)

    ftle_ = ftle.isel(time=0)
    ftle_ = (4 / timeseq.shape[0]) * np.log(ftle_)
    ftle_ = ftle_.sortby('longitude')
    ftle_ = ftle_.sortby('latitude')
    _, ridges = find_ridges_spherical_hessian(ftle_, sigma=1, scheme='second_order')
    ridges = ridges.where(ridges < -9e-10, 0)  # Warning: sensitive to sigma
    ridges = ridges.where(ridges >= -9e-10, 1)

    ftle_ = ftle_.interp(latitude=ridges.latitude, longitude=ridges.longitude)
    ridges = filter_ridges(ridges, ftle_, criteria=['mean_intensity', 'area'],thresholds=[1.5, 30])
    ridges = ridges.sortby('latitude')
    ridges = ridges.sortby('longitude')
    ftle_ = ftle_.sortby('longitude')
    ftle_ = ftle_.sortby('latitude')

    fig = plt.figure(figsize=[5, 15])
    spec = GS(4, 1, figure=fig, height_ratios=[0.32, 0.32, 0.32, 0.02])
    axs=[]
    axs.append(fig.add_subplot(spec[0], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[1], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[2], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[3]))

    x_departure.plot(transform=ccrs.PlateCarree(), ax=axs[0], add_colorbar=False, cmap=cmr.fusion, vmin=x_departure.min(),
                     vmax=x_departure.max(), antialiased=True, rasterized=True)
    y_departure.plot(transform=ccrs.PlateCarree(), ax=axs[1], add_colorbar=False, cmap=cmr.fusion_r,
                     vmin=y_departure.min(), vmax=y_departure.max(), antialiased=True, rasterized=True)
    p = ftle_.plot(transform=ccrs.PlateCarree(), ax=axs[2], add_colorbar=False, cmap=cmr.freeze,
                   antialiased=True, rasterized=True, vmin=0.5, vmax=2.5)
    ridges.plot(ax=axs[2], cmap=cmr.sunburst, alpha=.4, transform=ccrs.PlateCarree(),
                add_colorbar=False, edgecolors='none')

    cbar = fig.colorbar(p, cax=axs[3], orientation='horizontal', shrink=0.7)
    gl = axs[0].gridlines(draw_labels=True)

    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.xlines = False
    gl.ylines = False

    gl = axs[2].gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right = False
    gl.xlocator = ticker.FixedLocator([-80, -60, -40])
    gl.xlines = False
    gl.ylines = False

    cbar.ax.set_xlabel('FTLE (1/day)')
    axs[2].coastlines(color='gray')
    axs[1].coastlines(color='gray')
    axs[0].coastlines(color='gray')
    axs[2].set_title(' ',)
    axs[1].set_title(' ')
    axs[0].set_title(' ')
    axs[2].set_title('c', loc='left')
    axs[1].set_title('b', loc='left')
    axs[0].set_title('a', loc='left')
    plt.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.06)
    plt.savefig(f'tempfigs/local_example/chaotic_mixing_{ftle.time.values[0]}.png')

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[16, 8])
    p = tcwv.sel(time=ftle_.time.values).interp(latitude=ftle_.latitude, longitude=ftle_.longitude,
                                                method='linear').plot(
                                                                      ax=axs[0], cmap=cmr.gem, vmin=20, vmax=70,
                                                                      cbar_kwargs={'shrink': 0.8},
                                                                      transform=ccrs.PlateCarree())

    p.set_edgecolor('face')
    axs[0].coastlines(color='white')
    c = pr.sel(time=ftle_.time.values).interp(latitude=ftle_.latitude, longitude=ftle_.longitude,
                                                method='linear').plot.contour(ax=axs[0],
                                                                         levels=[1, 5, 10],
                                                                         cmap=cmr.freeze,
                                                                         add_colorbar=False,
                                                                         transform=ccrs.PlateCarree())
    # axs[0].set_title('FTLE')

    uplot = u.sel(time=ftle.time.values[0]).interp(latitude=ftle.latitude, longitude=ftle.longitude, method='linear')
    vplot = v.sel(time=ftle.time.values[0]).interp(latitude=ftle.latitude, longitude=ftle.longitude, method='linear')
    mag = np.sqrt(uplot**2 + vplot**2).values
    axs[0].streamplot(x=uplot.longitude.values, y=vplot.latitude.values, u=uplot.values,
              v=vplot.values, color='white', transform=ccrs.PlateCarree(), linewidth=mag/6)
    ridges.plot(ax=axs[0], cmap=cmr.sunburst, alpha=.8, transform=ccrs.PlateCarree(), add_colorbar=False)
    p = ftle_.plot(ax=axs[1], cbar_kwargs={'shrink': 0.8}, vmin=0.5, vmax=2.5, cmap=cmr.freeze,
                   transform=ccrs.PlateCarree())

    # for cl in p.collections: cl.set_edgecolor('face')
    p.set_edgecolor('face')

    axs[1].coastlines(color='white')

    plt.savefig(f'tempfigs/local_example/{ftle.time.values[0]}.png')
    plt.close()

    ftle_list.append(ftle)
    # lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
    #               return_det=False,cg_lambda=np.min)
    # ftle_min = lcs(ds.isel(time=timeseq))
    # ftle_min = np.sqrt(ftle_min)
    # potential_rainfall_MAS = (tcwv.isel(time=3) / (-6*3600*timeseq.shape[0])) * (1 - ftle)/ftle

    # potential_rainfall_MAS = - ( 2 * tcwv / (-6*3600*5) ) * (ftle - 1) / (ftle + 1)  # [mm / s]
    # potential_rainfall_MAS = potential_rainfall_MAS * 86400  # conversion to [mm / day]
print(1)
# ftle = xr.concat(ftle_list, 'time')
#
# conv = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')
# conv.isel(time=timeseq).mean('time').plot(vmin=-5, vmax=5, cmap=cmr.redshift)
# plt.show()
#
# rain=xr.open_dataarray('~/Downloads/rain.nc')
# rain = rain.assign_coords(longitude=(rain.coords['longitude'].values + 180) % 360 - 180)
# rain = rain.sortby('latitude')
# rain = rain.sortby('longitude')
# rain = rain.sum('time')
# rain = rain.interp(latitude=ftle.latitude, longitude=ftle.longitude, method='linear')
# rain.name = None
# tcwv.name = None
#
# fig, axs = plt.subplots(2, 3, subplot_kw={'projection': ccrs.PlateCarree()}, )
# ax = axs[0, 0]
# p = potential_rainfall.plot(cmap=cmr.rainforest, ax=ax, vmin=0, cbar_kwargs={'shrink': 0.8},
#                             transform=ccrs.PlateCarree())
# p.set_edgecolor('face')
# ax.coastlines(color='white')
# ax.set_title('Pot rain det')
#
# ax = axs[0, 1]
# p = potential_rainfall_MAS.plot(cmap=cmr.rainforest, ax=ax, vmin=0, cbar_kwargs={'shrink': 0.8},
#                                 transform=ccrs.PlateCarree())
# p.set_edgecolor('face')
# ax.coastlines(color='white')
# ax.set_title('Pot rain FTLE')
#
# ax = axs[0, 2]
# p =(timeseq.shape[0]**-1 * np.log(ftle)).where(det > 0).plot(cmap=cmr.rainforest, ax=ax, vmin=0, cbar_kwargs={'shrink': 0.8},
#                                              transform=ccrs.PlateCarree())
# p.set_edgecolor('face')
# ax.coastlines(color='white')
# ax.set_title('FTLE')
#
# ax = axs[1, 0]
# p = (rain*3600).plot(cmap=cmr.rainforest, ax=ax, vmax=40, cbar_kwargs={'shrink': 0.8}, transform=ccrs.PlateCarree())
# p.set_edgecolor('face')
# ax.coastlines(color='white')
# ax.set_title('ERA5 rain avg')
#
# ax = axs[1, 1]
# p = tcwv.isel(time=[0, 1, 2, 3]).mean('time').sel(latitude=ftle.latitude, longitude=ftle.longitude,
#                                             method='nearest').plot(cmap=cmr.rainforest, ax=ax, cbar_kwargs={'shrink': 0.8},
#                                                                    transform=ccrs.PlateCarree())
# p.set_edgecolor('face')
# ax.coastlines(color='white')
# ax.set_title('ERA5 tcwv avg')
#
# ax = axs[1, 2]
# p = (tcwv.isel(time=3) - tcwv.isel(time=0)).sel(latitude=ftle.latitude, longitude=ftle.longitude,
#                                             method='nearest').plot(cmap=cmr.redshift, ax=ax, cbar_kwargs={'shrink': 0.8},
#                                                                    transform=ccrs.PlateCarree())
# p.set_edgecolor('face')
# ax.coastlines(color='white')
# ax.set_title('ERA5 tcwv diff')
# plt.savefig('tempfigs/local_example.pdf')
# plt.show()



# for t in ftle.time.values:
#     ftle_ = ftle.sel(time=t)
#     ftle_ = ftle_.where(ftle_ > ftle.quantile(0.8), 1)
#     ridges = hessian(ftle_, black_ridges=True)
#
#     ridges = ftle_.copy(data=ridges)
#
#     fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8,8])
#     p = tcwv.sel(time=t).interp(latitude=ftle.latitude, longitude=ftle.longitude,
#                                                 method='linear').plot(ax=ax, cmap=cmr.gem, vmin=20, vmax=70)
#     p.set_edgecolor('face')
#     ax.coastlines(color='white')
#     ax.set_title('FTLE')
#
#     uplot = u.sel(time=t).interp(latitude=ftle.latitude, longitude= ftle.longitude, method='linear')
#     vplot = v.sel(time=t).interp(latitude=ftle.latitude, longitude= ftle.longitude, method='linear')
#
#     ax.streamplot(x=uplot.longitude.values, y=vplot.latitude.values, u=uplot.values,
#               v=vplot.values, color='white')
#     ridges.plot.contour(ax=ax,cmap='Reds')
#     plt.savefig(f'tempfigs/local_example/{t}.png')
#     plt.close()
