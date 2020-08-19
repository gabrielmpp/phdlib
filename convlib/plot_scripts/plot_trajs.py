import cartopy.crs as ccrs
import xarray as xr
from LagrangianCoherence.LCS import LCS
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from cartopy.io.img_tiles import Stamen
import pandas as pd
from owslib.wmts import WebMapTileService
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

# timesel = sys.argv[1]
timesel = slice('2020-01-20', '2020-01-28')
subdomain = {'latitude': slice(-40, 15),
             'longitude': slice(-85, -32)}
u = xr.open_dataarray(u_filepath, chunks={'time': 140})
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')


u = u.sel(latitude=slice(-40, 0), longitude=slice(-70, -35)).sel(expver=1).drop('expver')
u = u.sel(time=timesel)
u = u.load()

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-40, 0), longitude=slice(-70, -35)).sel(expver=1).drop('expver')
v = v.sel(time=timesel)
v = v.load()

tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-40, 0), longitude=slice(-70, -35)).sel(expver=1).drop('expver')
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
    [-14.3, -55],
    [-20.7, -56],
    [-24.1, -52],
]

for dt in range(0, ntimes):
    pts_x_list = []
    pts_y_list = []

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_xlim([-60, -38])
    ax.set_ylim([-30, -10])
    ax.coastlines('10m')
    timeseq = np.arange(0, 8) + dt
    ds = xr.merge([u, v])
    ds = ds.isel(time=timeseq)
    ds = ds.resample(time='15T').interpolate('linear')
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)
    ax.background_img('BM', resolution='low')
    ax.add_feature(states_provinces, edgecolor='gray')

    ax.set_title('Smoke trajectories at ' + pd.Timestamp(ds.time.values[-1]).strftime('%d-%m-%Y'),
                 x=0.5, y=0.9, color='white')

    xs, ys = parcel_propagation(ds.u, ds.v, timestep=900,
                                propdim='time', SETTLS_order=3, return_traj=True)

    for pt in pts:
        pts_x = xs.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        pts_y = ys.sel(latitude=pt[0], longitude=pt[1], method='nearest')
        pts_x_list.append(pts_x)
        pts_y_list.append(pts_y)
    pts_x = np.column_stack(pts_x_list)
    pts_y = np.column_stack(pts_y_list)
    colors = np.arange(len(pts_x))/len(pts_x)
    colors= np.column_stack([colors,colors, colors])
    p = ax.scatter(pts_x, pts_y, transform=ccrs.PlateCarree(),  cmap=cmr.sunburst_r, c=colors)
    ax.scatter(pts_x[0,:], pts_y[0,:], marker='^', color='red')
    plt.savefig('../tempfigs/anim_trajs/{:03d}'.format(dt), transparent=True)
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