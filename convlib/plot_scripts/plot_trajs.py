import cartopy.crs as ccrs
import xarray as xr
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


u = u.sel(latitude=slice(-60, 60), longitude=slice(-130, 45)).sel(expver=1).drop('expver')
u = u.sel(time=timesel)
u = u.load()

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-60, 60), longitude=slice(-130, 45)).sel(expver=1).drop('expver')
v = v.sel(time=timesel)
v = v.load()

tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-60, 60), longitude=slice(-130, 45)).sel(expver=1).drop('expver')
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


u_ = u.isel(time=2).coarsen(latitude=5, longitude=5, boundary='trim').mean()
v_ = v.isel(time=2).coarsen(latitude=5, longitude=5, boundary='trim').mean()
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
pts = [
    [-13, -47],
    [-13, -48],
    [-14, -48],
    [-14, -47],
]

for dt in range(0, ntimes):



    pts_x_list = []
    pts_y_list = []
    dt=12

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
    ftle = lcs(ds, s=1e5, resample='3H')
    ftle = np.log(ftle) / 2
    u11 = ds.u.diff('longitude')
    u12 = ds.u.diff('latitude')
    u21 = ds.v.diff('longitude')
    u22 = ds.v.diff('latitude')
    S = .5 * (u12 + u21)
    AS = .5 * (u12 - u21)
    Q = .5 * (AS **2 - S ** 2)


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    # ax.set_xlim([-70, -32])
    # ax.set_ylim([-40, -5])
    ax.coastlines('10m', color='gray', linewidth=2)
    ftle.plot(vmax=2.5, vmin=.5, cmap=cmr.freeze, ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)
    ax.set_title('')
    plt.savefig('../tempfigs/fig3_for_scheme.png', dpi=600,
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    ftle = ftle.isel(time=0)

    ridges, _ = find_ridges_spherical_hessian(ftle,
                                              sigma=1.2,
                                              scheme='second_order',
                                              angle=15,
                                              ds_wind=ds.isel(time=-1).sel(latitude=ftle.latitude,
                                                             longitude=ftle.longitude,
                                                             method='nearest'))
    valleys, _ = find_ridges_spherical_hessian(-ftle,
                                              sigma=1.2,
                                              scheme='second_order',
                                              angle=15,
                                              ds_wind=ds.isel(time=-1).sel(latitude=ftle.latitude,
                                                             longitude=ftle.longitude,
                                                             method='nearest'))
    ridges.plot()
    plt.show()
    # ridges = ridges.where(ridges < -9e-10, 0)  # Warning: sensitive to sigma
    # ridges = ridges.where(ridges >= -9e-10, 1)

    ftle = ftle.interp(latitude=ridges.latitude, longitude=ridges.longitude)
    valleys = valleys.interp(latitude=ftle.latitude, longitude=ftle.longitude)

    ridges = filter_ridges(ridges, ftle, criteria=['mean_intensity', 'major_axis_length'],
                           thresholds=[1.2, 20])
    valleys = filter_ridges(valleys, -ftle, criteria=[ 'major_axis_length'],
                           thresholds=[ 20])
    ridges_ = ridges.where(~xr.ufuncs.isnan(ridges), 0)
    valleys_ = ridges.where(~xr.ufuncs.isnan(valleys), 0)
    from skimage.morphology import watershed
    from scipy import ndimage
    s = ndimage.generate_binary_structure(2, 2)  # Full connectivity in space

    markers_ridges = ndimage.label(ridges_.values, structure=s)[0]
    markers_valleys = ndimage.label(valleys_.values, structure=s)[0]
    labels_ridges = watershed(ndimage.gaussian_filter(-ftle.values, sigma=1), markers=markers_ridges)
    labels_valleys = watershed(ndimage.gaussian_filter(ftle.values, sigma=1), markers=markers_valleys)
    labels_ridges = ftle.copy(data=labels_ridges)
    labels_valleys = ftle.copy(data=labels_valleys)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})

    labels_ridges.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='tab20')
    # labels.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='k')
    ridges.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)

    #   HERE TRYING TO TAKE WIND ANOMALY
    # u_ = ds.u.copy()
    # u_ = u_ - u_.mean('longitude') - u_.mean('latitude')
    # v_ = ds.v.copy()
    # v_ = v_ - v_.mean('longitude')- v_.mean('latitude')
    u_ = ds.u
    v_ = ds.v
    u_ = u_.isel(time=-1).sel(latitude=ftle.latitude, longitude=ftle.longitude,
                                method='nearest').coarsen(latitude=5, longitude=5, boundary='trim').mean()

    v_ = v_.isel(time=-1).sel(latitude=ftle.latitude, longitude=ftle.longitude,
                                method='nearest').coarsen(latitude=5, longitude=5, boundary='trim').mean()
    ax.quiver(u_.longitude.values, v_.latitude.values,
              u_.values, v_.values)
    ax.coastlines()
    plt.savefig('../tempfigs/watersheds1.png', dpi=600,
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})

    labels_valleys.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='tab20')
    # labels.plot.contour(ax=ax, transform=ccrs.PlateCarree(), colors='k')
    ridges.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False)

    #   HERE TRYING TO TAKE WIND ANOMALY
    # u_ = ds.u - ds_seasonal_avgs.u_season.sel(season='DJF')
    # v_ = ds.v - ds_seasonal_avgs.v_season.sel(season='DJF')
    u_ = ds.u
    v_ = ds.v
    u_ = u_.isel(time=-1).sel(latitude=ftle.latitude, longitude=ftle.longitude,
                                method='nearest').coarsen(latitude=5, longitude=5, boundary='trim').mean()

    v_ = v_.isel(time=-1).sel(latitude=ftle.latitude, longitude=ftle.longitude,
                                method='nearest').coarsen(latitude=5, longitude=5, boundary='trim').mean()
    ax.quiver(u_.longitude.values, v_.latitude.values,
              u_.values, v_.values)
    ax.coastlines()
    plt.savefig('../tempfigs/watersheds2.png', dpi=600,
                transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
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

    ds = ds.resample(time='2H').interpolate('linear')

    xs, ys = parcel_propagation(ds.u, ds.v, timestep=-3600 * 2,
                            propdim='time', SETTLS_order=5, return_traj=True)

    xs_ = xs.where(ridges==1)
    ys_ = ys.where(ridges==1)
    xs.isel(time=-1).plot()
    plt.show()
    from shapely.geometry import polygon
    pts = [
        xs_.stack({'points':['latitude', 'longitude']}).dropna('points').values,
        ys_.stack({'points':['latitude', 'longitude']}).dropna('points').values,
    ]
    for pt in pts:
        pts_x = xs.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        pts_y = ys.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        pts_x_list.append(pts_x)
        pts_y_list.append(pts_y)

    pts_x = np.column_stack(pts_x_list)
    pts_y = np.column_stack(pts_y_list)
    colors = np.arange(len(pts_x))/len(pts_x)
    colors = np.column_stack([colors for _ in range(8) for __ in range(pts_x.shape[-1])])
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_xlim([-60, -38])
    ax.set_ylim([-30, -10])
    ax.coastlines('10m')
    p = ax.scatter(pts_x, pts_y, transform=ccrs.PlateCarree(),  cmap='viridis', c=colors)
    ax.scatter(pts_x[-1,:], pts_y[-1,:], marker='x', color='red')
    plt.savefig('../tempfigs/anim_trajs/{:03d}'.format(dt), transparent=True, bbox_inches='tight',
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