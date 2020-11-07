from LagrangianCoherence.LCS.area_of_influence import find_area
import xarray as xr
import numpy as np
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib
import time
import cmasher as cmr
from skimage.morphology import skeletonize, binary_erosion, binary_dilation
from skimage.filters import threshold_local
from scipy import ndimage as ndi
import sys

event_subtropical_cyclone = slice('2020-06-25', '2020-07-05')
event_ITCZ = slice('2020-04-10', '2020-04-18')
event_ITCZ2 = slice('2020-03-12', '2020-03-18')
event_jbn = slice('2020-05-07', '2020-05-18')
casestudy = str(sys.argv[1])

if casestudy == 'itcz':
    event = event_ITCZ
elif casestudy == 'itcz2':
    event = event_ITCZ2
elif casestudy =='subtropical':
    event = event_subtropical_cyclone
elif casestudy == 'jbn':
    event = event_jbn
    
experiment_2days_path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/' \
                  'experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4'

data_basepath = '/home/gab/phd/data/ERA5/'
case_study_outpath = '/home/gab/phd/data/case_studies/'

u_filename = 'ERA5viwve_ERA5_6hr_{year}010100-{year}123118.nc'
v_filename = 'ERA5viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
tcwv_filename = 'ERA5tcwv_ERA5_6hr_{year}010100-{year}123118.nc'

# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
sfcpres_filepath = basepath + 'mslpres_ERA5_6hr_2020010100-2020123118.nc'
orog_filepath = basepath + 'geopotential_orography.nc'
data_dir = '/home/gab/phd/data/composites_cz/'
pr_filepath = basepath + 'pr_ERA5_6hr_2020010100-2020123118.nc'

subdomain = {'latitude': slice(-40, 15),
             'longitude': slice(-90, -32)}
u = xr.open_dataarray(u_filepath, chunks={'time': 140})
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')

u = u.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')

tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')

sfcpres = xr.open_dataarray(sfcpres_filepath, chunks={'time': 140})
sfcpres = sfcpres.assign_coords(longitude=(sfcpres.coords['longitude'].values + 180) % 360 - 180)
sfcpres = sfcpres.sortby('longitude')
sfcpres = sfcpres.sortby('latitude')
sfcpres = sfcpres.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')

pr = xr.open_dataarray(pr_filepath, chunks={'time': 140})
pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
pr = pr.sortby('longitude')
pr = pr.sortby('latitude')
pr = pr.sel(latitude=slice(-75, 60), longitude=slice(-150, 45))
pr = pr * 3600



u = u / tcwv
v = v / tcwv
u.name = 'u'
v.name = 'v'



timesel = event

u = u.sel(time=timesel)
v = v.sel(time=timesel)
tcwv = tcwv.sel(time=timesel)
sfcpres = sfcpres.sel(time=timesel)
pr = pr.sel(time=timesel).sel(expver=1).drop('expver')
pr = pr.load()
u = u.load()
sfcpres = sfcpres.load()
tcwv = tcwv.load()
v = v.load()

for dt in range(1, u.time.values.shape[0]):
    time1 = time.time()
    timeseq = np.arange(0, 8) + dt
    ds = xr.merge([u, v])
    ds = ds.isel(time=timeseq)
    lcs = LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, gauss_sigma=None)
    ftle = lcs(ds, s=1e5, resample='3H')
    lcs_local = LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, gauss_sigma=None)
    ftle_local = lcs(ds.isel(time=slice(-1, None)), s=1e5, resample=None)
    ftle = np.log(ftle) / 2
    ftle_local = np.log(ftle_local) * 4
    ftle_local = ftle_local.isel(time=0)
    ftle = ftle.isel(time=0)

    block_size = 301
    local_thresh = ftle_local.copy(data=threshold_local(ftle_local.values, block_size,
                                                        offset=-.8))
    binary_local = ftle_local > local_thresh
    ftle_local_high = binary_local

    distance = binary_local
    ridges, eigmin, eigvectors = find_ridges_spherical_hessian(ftle,
                                                               sigma=1.2,
                                                               scheme='second_order',
                                                               angle=20 , return_eigvectors=True)

    ridges = ridges.copy(data=skeletonize(ridges.values))

    ridges = filter_ridges(ridges, ftle, criteria=['mean_intensity', 'major_axis_length'],
                           thresholds=[1.2, 30 ])

    ridges = ridges.where(~xr.ufuncs.isnan(ridges), 0)

    sfcpres = sfcpres.interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    dpdx = sfcpres.isel(time=-1).differentiate('longitude')
    dpdy = sfcpres.isel(time=-1).differentiate('latitude')
    u_vec = eigvectors.isel(eigvectors=1)
    v_vec = eigvectors.isel(eigvectors=0)


    pres_gradient_parallel = np.sqrt((dpdx * v_vec) ** 2 + (dpdy * u_vec) ** 2)
    ridges_pres_grad = ridges * pres_gradient_parallel

    ridges_pres_grad = filter_ridges(ridges, ridges_pres_grad, criteria=['mean_intensity'],
                                     thresholds=[50])
    ridges_bool = ridges == 1
    dist = ftle_local_high.copy(data=ndi.distance_transform_edt(~ridges_bool))

    ridges_dil = ridges.copy(data=binary_dilation(ridges.values))
    ridges_dil = ridges_dil.where(ridges_dil > 0)
    ridges = ridges.where(ridges > 0)
    ridges_pres_grad = ridges_pres_grad.where(ridges_pres_grad > 0)
    ridges_pres_grad = ridges_pres_grad.where(xr.ufuncs.isnan(ridges_pres_grad), 1)

    ridges_ = filter_ridges(ftle_local_high, ridges_dil.where(~xr.ufuncs.isnan(ridges_dil), 0),
                            criteria=['max_intensity'],
                            thresholds=[.5])
    ridges_ = ridges_ * dist.where(dist < 12)
    ridges_ = ridges_.where(xr.ufuncs.isnan(ridges_), 1)
    local_strain = ftle_local_high - ridges_.where(~xr.ufuncs.isnan(ridges_), 0)

    pr_ = pr.sel(time=ridges.time).interp(method='nearest',
                                              latitude=ridges.latitude,
                                              longitude=ridges.longitude)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    sfcpres.sel(time=ridges.time).plot(cmap='viridis', ax=ax,
                                                   linewidth=.5,
                                                   add_colorbar=False)

    pr_.plot.contourf(add_colorbar=False, levels=[1, 5, 10, 15], alpha=.5, cmap=cmr.freeze_r, ax=ax)
    matplotlib.rcParams['hatch.color'] = 'blue'
    matplotlib.rcParams['hatch.linewidth'] = .3

    local_strain.plot.contourf(cmap='Reds', alpha=0,
                                   levels=[0, .5],
                                   add_colorbar=False,
                                   hatches=['',
                                            '////////'],
                                   ax=ax)
    matplotlib.rcParams['hatch.color'] = 'red'
    ridges_.where(~xr.ufuncs.isnan(ridges_), 0).plot.contourf(cmap='Reds', alpha=0,
                                                                  levels=[0, .5],
                                                                  add_colorbar=False,
                                                                  hatches=['',
                                                                           'xxxxxxx'],
                                                                  ax=ax)
    ridges.plot(add_colorbar=False, cmap='Purples', ax=ax)
    ridges_pres_grad.plot(add_colorbar=False, cmap=cmr.bubblegum, ax=ax)
    ax.coastlines(color='black')
    total_rain = np.sum(pr_)
    czs_rain = np.sum(ridges_ * pr_)
    local_strain_rain = np.sum(local_strain * pr_)
    rest = total_rain - local_strain_rain - czs_rain
    ax.text(-90, -35, str(np.round(czs_rain.values / 1000)) + ' m rain on CZs \n ' +
                str(np.round(local_strain_rain.values / 1000)) + ' m rain on LStr \n' +
                str(np.round(rest.values / 1000)) + 'm remaining \n',
                bbox={'color': 'black', 'alpha': .2},
                color='black')
    plt.savefig(f'case_studies/{casestudy}/fig{dt:02d}_area.png', dpi=600,
                    transparent=False, pad_inches=.2, bbox_inches='tight'
                    )
    plt.close()

    time2 = time.time()
    ellapsed = time2 - time1
    print('Ellapsed time: ' + str(np.round(ellapsed / 60)) + 'minutes')

