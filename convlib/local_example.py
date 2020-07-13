import cartopy.crs as ccrs
import xarray as xr
from LagrangianCoherence.LCS import LCS
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
subdomain = {'latitude': slice(-40, -10),
             'longitude': slice(-60, -30)}
u = xr.open_dataarray(u_filepath)
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')

u = u.sel(latitude=slice(-70, 10), longitude=slice(-110, -1))

v = xr.open_dataarray(v_filepath)
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-70, 10), longitude=slice(-110, -1))

tcwv = xr.open_dataarray(tcwv_filepath)
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-70, 10), longitude=slice(-110, -1))
rain = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')

u = u/tcwv
v = v/tcwv
u.name = 'u'
v.name = 'v'
ds = xr.merge([u, v])

# ---- Running LCS ---- #

lcs = LCS.LCS(lcs_type='attracting', timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
              return_det=True)
det = lcs(ds.isel(time=[0, 1, 2, 3]))
det = np.sqrt(det)
potential_rainfall = - ( 2 * tcwv / (-6*3600*5) ) * (det - 1) / (det + 1)  # [mm / s]
potential_rainfall = potential_rainfall * 86400  # conversion to [mm / day]

lcs = LCS.LCS(lcs_type='attracting', timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
              return_det=False)
ftle = lcs(ds.isel(time=[0, 1, 2, 3]))
ftle = np.sqrt(ftle)

lcs = LCS.LCS(lcs_type='attracting', timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
              return_det=False,cg_lambda=np.min)
ftle_min = lcs(ds.isel(time=[0, 1, 2, 3]))
ftle_min = np.sqrt(ftle_min)

potential_rainfall_MAS = - ( 2 * tcwv / (-6*3600*5) ) * (ftle - 1) / (ftle + 1)  # [mm / s]
potential_rainfall_MAS = potential_rainfall_MAS * 86400  # conversion to [mm / day]


conv = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')
conv.isel(time=[0, 1, 2, 3]).mean('time').plot(vmin=-5, vmax=5, cmap=cmr.redshift)
plt.show()

det.plot(vmin=-5, vmax=5, cmap=cmr.redshift)
plt.show()
ftle.where(det>1).plot(vmin=0, vmax=5, cmap=cmr.rainforest)
plt.show()
rain=xr.open_dataarray('~/Downloads/rain.nc')
rain = rain.assign_coords(longitude=(rain.coords['longitude'].values + 180) % 360 - 180)
rain = rain.sortby('latitude')
rain = rain.sortby('longitude')
rain = rain.sum('time')
rain = rain.sel(latitude=potential_rainfall.latitude, longitude=potential_rainfall.longitude)
fig, ax= plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
potential_rainfall.plot(cmap=cmr.rainforest, ax=ax, vmin=30, vmax=80)
ax.coastlines(color='white')
plt.show()
fig, ax= plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
potential_rainfall_MAS.plot(cmap=cmr.rainforest, ax=ax, vmin=30, vmax=80)
ax.coastlines(color='white')
plt.show()
fig, ax= plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
ftle.where(det > 0).plot(cmap=cmr.rainforest, ax=ax, vmin=0, vmax=80)
ax.coastlines(color='white')
plt.show()
fig, ax= plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
(rain*3600).plot(cmap=cmr.rainforest, ax=ax, vmax=40)
ax.coastlines(color='white')
plt.show()
fig, ax= plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
(rain*3600).plot(cmap=cmr.rainforest, ax=ax, vmax=40)
ax.coastlines(color='white')
plt.show()  
