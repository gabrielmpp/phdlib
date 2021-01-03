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
from LagrangianCoherence.LCS.LCS import compute_deftensor
from numpy.linalg import norm
import sys
from skimage.filters import gaussian

def to_potential_temp(tmp, pres):
    return tmp * (100000 / pres) ** 0.286



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
tmp_filepath = basepath + 'tmp_2m_ERA5_6hr_2020010100-2020123118.nc'
mslpres_filepath = basepath + 'mslpres_ERA5_6hr_2020010100-2020123118.nc'
sfcpres_filepath = basepath + 'sfcpres_ERA5_6hr_2020010100-2020123118.nc'
orog_filepath = basepath + 'geopotential_orography.nc'

da = xr.open_dataarray(orog_filepath)
da = da.assign_coords(longitude=(da.coords['longitude'].values + 180) % 360 - 180)
da = da.sortby('longitude')
da = da.sortby('latitude')
da = da.sel(latitude=slice(-65, 20), longitude=slice(-100, -25)).isel(time=0).drop('time')

sfcpres = xr.open_dataarray(sfcpres_filepath, chunks={'time': 140})
sfcpres = sfcpres.assign_coords(longitude=(sfcpres.coords['longitude'].values + 180) % 360 - 180)
sfcpres = sfcpres.sortby('longitude')
sfcpres = sfcpres.sortby('latitude')
sfcpres = sfcpres.sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1, time='2020-06-27T00:00:00').drop('expver')


tmp = xr.open_dataset(tmp_filepath, chunks={'time': 140})['t2m_0001']
tmp = tmp.assign_coords(longitude=(tmp.coords['longitude'].values + 180) % 360 - 180)
tmp = tmp.sortby('longitude')
tmp = tmp.sortby('latitude')
tmp = tmp.sel(latitude=slice(-65, 20), longitude=slice(-100, -25), time='2020-06-27T00:00:00')
tmp = to_potential_temp(tmp, sfcpres)
tmp = tmp.load()
tmp.plot()
plt.show()

def_tensor = compute_deftensor(da, da, sigma=1.5)
def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
def_tensor = def_tensor.dropna(dim='points')

vals = def_tensor.transpose(..., 'points').values
vals = vals.reshape([2, 2, def_tensor.shape[-1]])
def_tensor_norm_tmp = norm(vals, axis=(0, 1), ord=2)
def_tensor_norm_tmp = def_tensor.isel(derivatives=0).drop('derivatives').copy(data=def_tensor_norm_tmp)
def_tensor_norm_tmp = def_tensor_norm_tmp.unstack('points')
def_tensor_norm_tmp.plot()
plt.show()
ridges_tmp, eivals_temp, eigvectors_tmp = find_ridges_spherical_hessian(da,
                                                                        sigma=1.2,
                                                                        scheme='second_order',
                                                                        angle=50, return_eigvectors=True)
u_tmp = eigvectors_tmp.isel(eigvectors=1) * da/30
v_tmp = eigvectors_tmp.isel(eigvectors=0) * da/30
u_tmp_lowres = u_tmp.coarsen(latitude=5, longitude=5, boundary='trim').mean()
v_tmp_lowres = v_tmp.coarsen(latitude=5, longitude=5, boundary='trim').mean()

fig, axs = plt.subplots(1, 2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        )
da.plot(ax=axs[0], transform=ccrs.PlateCarree(), add_colorbar=False, robust=True)
da.plot(ax=axs[1], add_colorbar=False, robust=True)
axs[1].quiver(u_tmp_lowres.longitude.values,
           u_tmp_lowres.latitude.values,
           u_tmp_lowres.values, v_tmp_lowres.values)
axs[0].coastlines()
axs[1].coastlines()
plt.savefig(f'case_studies/topo.png', dpi=600,
            transparent=False, pad_inches=.2, bbox_inches='tight'
            )
plt.close()





def_tensor = compute_deftensor(tmp, tmp, sigma=1.5)
def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
def_tensor = def_tensor.dropna(dim='points')

vals = def_tensor.transpose(..., 'points').values
vals = vals.reshape([2, 2, def_tensor.shape[-1]])
def_tensor_norm_tmp = norm(vals, axis=(0, 1), ord=2)
def_tensor_norm_tmp = def_tensor.isel(derivatives=0).drop('derivatives').copy(data=def_tensor_norm_tmp)
def_tensor_norm_tmp = def_tensor_norm_tmp.unstack('points')
def_tensor_norm_tmp.plot()
plt.show()
ridges_tmp, eivals_temp, eigvectors_tmp = find_ridges_spherical_hessian(def_tensor_norm_tmp,
                                                                        sigma=1.2,
                                                                        scheme='second_order',
                                                                        angle=50, return_eigvectors=True)
u_tmp = eigvectors_tmp.isel(eigvectors=1) * def_tensor_norm_tmp/30
v_tmp = eigvectors_tmp.isel(eigvectors=0) * def_tensor_norm_tmp/30
u_tmp_lowres = u_tmp.coarsen(latitude=5, longitude=5, boundary='trim').mean()
v_tmp_lowres = v_tmp.coarsen(latitude=5, longitude=5, boundary='trim').mean()

fig, axs = plt.subplots(1, 2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        )
tmp.plot(ax=axs[0], transform=ccrs.PlateCarree(), add_colorbar=False, robust=True, cmap=cmr.pride)
def_tensor_norm_tmp.plot(ax=axs[1], add_colorbar=False, robust=True)
axs[1].quiver(u_tmp_lowres.longitude.values,
           u_tmp_lowres.latitude.values,
           u_tmp_lowres.values, v_tmp_lowres.values)
axs[0].coastlines()
axs[1].coastlines()
plt.savefig(f'case_studies/tmp.png', dpi=600,
            transparent=False, pad_inches=.2, bbox_inches='tight'
            )
plt.close()