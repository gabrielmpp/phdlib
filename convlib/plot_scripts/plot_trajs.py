import cartopy.crs as ccrs
import xarray as xr
from skimage.morphology import skeletonize, binary_erosion, binary_dilation

from LagrangianCoherence.LCS import LCS
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from cartopy.io.img_tiles import Stamen
import pandas as pd
from owslib.wmts import WebMapTileService
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
from LagrangianCoherence.LCS.trajectory import parcel_propagation
from matplotlib.gridspec import GridSpec as GS
import sys
import cartopy.feature as cfeature
# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
pr_filepath = basepath + 'pr_ERA5_6hr_2020010100-2020123118.nc'
data_dir = '/home/gab/phd/data/composites_cz/'
ds_seasonal_avgs = xr.open_dataset(data_dir + 'ds_seasonal_avgs.nc')

# timesel = sys.argv[1]
timesel = slice('2020-01-20', '2020-02-28')
subdomain = {'latitude': slice(-40, 15),
             'longitude': slice(-90, -32)}
u = xr.open_dataarray(u_filepath, chunks={'time': 140})
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')
u = u.sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
u = u.sel(time=timesel)
u = u.load()

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
v = v.sel(time=timesel)
v = v.load()

tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
tcwv = tcwv.sel(time=timesel)
tcwv = tcwv.load()

pr = xr.open_dataarray(pr_filepath, chunks={'time': 140})
pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
pr = pr.sortby('longitude')
pr = pr.sortby('latitude')
pr = pr.sel(subdomain)
pr = pr.sel(time=timesel).isel(expver=0).drop('expver')
pr = pr.load()
pr = pr * 3600  # Originally in metres
pr.isel(time=-1).plot(cmap=cmr.freeze)
plt.show()
# rain = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')


u_ = u.isel(time=2).coarsen(latitude=15, longitude=15, boundary='trim').mean()
v_ = v.isel(time=2).coarsen(latitude=15, longitude=15, boundary='trim').mean()
tcwv_ = tcwv.isel(time=2)
from mpl_toolkits.mplot3d import Axes3D

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()},
                        gridspec_kw={'wspace': .05})
axs[0].quiver(
    u_.longitude.values,
    u_.latitude.values,
    u_.values,
    v_.values,
    transform=ccrs.PlateCarree()
)
axs[0].coastlines()
tcwv_.plot(ax=axs[1],add_colorbar=False, cmap=cmr.gem)
axs[1].coastlines()
axs[1].set_title('')
plt.savefig('../tempfigs/fig1_for_scheme.png', dpi=600,
            transparent=True, bbox_inches='tight', pad_inches=0)

plt.close()

u_ = u_/tcwv_
v_ = v_/tcwv_
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()},
                        gridspec_kw={'wspace': .05})
ax.quiver(
    u_.longitude.values,
    u_.latitude.values,
    u_.values,
    v_.values,
    transform=ccrs.PlateCarree()
)
ax.coastlines()

plt.savefig('../tempfigs/fig2_for_scheme.png', dpi=600,
            transparent=True, bbox_inches='tight', pad_inches=0)

plt.close()

u = u/tcwv
v = v/tcwv
u.name = 'u'
v.name = 'v'

###############
import os
os.environ["CARTOPY_USER_BACKGROUNDS"] = "/home/gab/phd/scripts/phdlib/convlib/plot_scripts/"
ntimes = u.time.values.shape[0]
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
base_url = 'https://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
layer_name = 'VIIRS_CityLights_2012'
land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
pts1 = [
    [-8, -44],
    [-8, -42],
    [-6, -44],
    [-6, -42],
]
pts2 = [
    [-10., -65],
    [-10., -63],
    [-8., -65],
    [-8., -63],
]
-10, -8
-50, -48

for dt in range(0, ntimes):




    timeseq = np.arange(0, 8) + dt
    ds = xr.merge([u, v])
    ds = ds.isel(time=timeseq)
    # ax.add_feature(cfeature.BORDERS)
    # ax.add_feature(cfeature.OCEAN)
    # ax.background_img('BM', resolution='low')
    # ax.add_feature(states_provinces, edgecolor='gray')
    #
    # ax.set_title('Smoke trajectories at ' + pd.Timestamp(ds.time.values[-1]).strftime('%d-%m-%Y'),
    #              x=0.5, y=0.9, color='white')
    lcs = LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, gauss_sigma=None)
    ftle = lcs(ds, s=1.6e6)
    ftle = np.log(ftle) / 2
    ftle = ftle.isel(time=0)

    ridges, eigmin, eigvectors = find_ridges_spherical_hessian(ftle,
                                                               sigma=None,
                                                               scheme='second_order',
                                                               angle=15, return_eigvectors=True)

    ridges = ridges.copy(data=skeletonize(ridges.values))

    ridges = filter_ridges(ridges, ftle, criteria=['mean_intensity', 'major_axis_length'],
                           thresholds=[1, 30 ])


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    # ax.set_xlim([-70, -32])
    # ax.set_ylim([-40, -5])
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER

    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    gl.xlabels_right = False
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.xlines = False
    gl.ylines = False


    ax.coastlines('10m', color='gray', linewidth=2)
    ftle.plot(vmax=2.5, vmin=.5, cmap=cmr.freeze, ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)
    ax.set_title('')
    plt.savefig('../tempfigs/fig3_for_scheme.png', dpi=600,
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()


    ridges, eigmin, eigvectors = find_ridges_spherical_hessian(ftle,
                                                               sigma=None,
                                                               scheme='second_order',
                                                               angle=15, return_eigvectors=True)

    ridges = ridges.copy(data=skeletonize(ridges.values))

    ridges = filter_ridges(ridges, ftle, criteria=['mean_intensity', 'major_axis_length'],
                           thresholds=[1, 30])
    ridges.plot()
    plt.show()
    # ridges = ridges.where(ridges < -9e-10, 0)  # Warning: sensitive to sigma
    # ridges = ridges.where(ridges >= -9e-10, 1)

    ftle = ftle.interp(latitude=ridges.latitude, longitude=ridges.longitude)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER

    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    gl.xlabels_right = False
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.xlines = False
    gl.ylines = False

    # ax.set_xlim([-70, -32])
    # ax.set_ylim([-40, -5])
    ax.coastlines('10m', color='gray', linewidth=2)
    ftle.plot(vmax=2.5, vmin=.5, cmap=cmr.freeze, ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)
    ridges.plot(ax=ax, cmap=cmr.sunburst, alpha=.8, transform=ccrs.PlateCarree(),
                add_colorbar=False, edgecolors='none')
    ax.set_title('')
    plt.savefig('../tempfigs/fig4_for_scheme.png', dpi=600,
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()


    xs, ys = parcel_propagation(ds.u, ds.v, timestep=-3600 * 6, s=1.3e6,
                            propdim='time', SETTLS_order=5, return_traj=True)

    xs_ = xs.where(ridges==1)
    ys_ = ys.where(ridges==1)
    xs_.isel(time=-1).plot()
    plt.show()
    from shapely.geometry import polygon


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER

    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    gl.xlabels_right = False
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.xlines = False
    gl.ylines = False

    # ax.set_xlim([-90, -20])
    # ax.set_ylim([-40, 10])
    for pt in pts1:
        pts_x = xs.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        pts_y = ys.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        plt.plot(pts_x.values, pts_y.values, linestyle='--', color='blue')
        ax.plot(pts_x.values[-1], pts_y.values[-1], marker='x', color='red')
    for pt in pts2:
        pts_x = xs.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        pts_y = ys.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        plt.plot(pts_x.values, pts_y.values, linestyle='--', color='green')
        ax.plot(pts_x.values[-1], pts_y.values[-1], marker='x', color='red')
    # ftle.plot(vmax=2.5, vmin=.5, cmap=cmr.freeze, ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, alpha=1)
    # ridges.plot(ax=ax, cmap=cmr.sunburst, alpha=.8, transform=ccrs.PlateCarree(),
    #             add_colorbar=False, edgecolors='none')
    ax.coastlines('10m', color='gray', linewidth=2)
    plt.savefig('../tempfigs/fig_traj_scheme.png', transparent=True, bbox_inches='tight',
                pad_inches=0, dpi=600)
    plt.close()



fig.colorbar(p)


xs.name = 'x'
ys.name = 'y'
concentration = xs.copy(data=np.ones(xs.values.shape))
data = np.onse(xs.values.shape)
for i in range(concentration.latitude.values):
    for j in range(concentration.longitude.values):
        data[i, j] = np.random.normal()
ds_traj = xr.merge([xs, ys])
a = ds_traj.where((ds_traj.x==-50.5) & (ds_traj.y == -24.5), 0)
a.x.isel(time=8).plot()
plt.show()