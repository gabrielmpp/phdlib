import cartopy.crs as ccrs
import xarray as xr
from LagrangianCoherence.LCS import LCS
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr
from xr_tools.tools import find_ridges_spherical_hessian

from skimage.filters import hessian, frangi, sato
import pandas as pd
# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
subdomain = {'latitude': slice(-40, -5),
             'longitude': slice(-65, -30)}
u = xr.open_dataarray(u_filepath)
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')

u = u.sel(latitude=slice(-70, 30), longitude=slice(-130, 30))

v = xr.open_dataarray(v_filepath)
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-70, 30), longitude=slice(-130, 30))

tcwv = xr.open_dataarray(tcwv_filepath)
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-70, 30), longitude=slice(-130, 30))
rain = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')

u = u/tcwv
v = v/tcwv
u.name = 'u'
v.name = 'v'
ds = xr.merge([u, v])

# ---- Running LCS ---- #
ntimes = 30*4
ftle_list = []

from scipy.ndimage import label, generate_binary_structure
from skimage.measure import regionprops_table


def filter_ridges(ridges, ftle, mean_intensity, min_area):
    s = generate_binary_structure(2, 2)  # Full connectivity
    ridges_labels=ridges.copy(data=label(ridges, structure=s)[0])
    df = pd.DataFrame(regionprops_table(ridges_labels.values,
                                        intensity_image=ftle.values,
                      properties=['label', 'area', 'mean_intensity']))
    df = df.set_index('label')

    # Area criterion
    da_area = df.to_xarray()['area']
    da_area = da_area.where(da_area > min_area, drop=True)
    criterion_1 = ridges_labels.copy()
    for l in da_area.label.values:
        criterion_1 = criterion_1.where(criterion_1 != l, 1)
    criterion_1 = criterion_1.where(criterion_1 == 1, 0)

    # Intensity criterion
    da_intensity = df.to_xarray()['mean_intensity']
    da_intensity = da_intensity.where(da_intensity > mean_intensity, drop=True)
    criterion_2 = ridges_labels.copy()
    for l in da_intensity.label.values:
        criterion_2 = criterion_2.where(criterion_2 != l, 1)
    criterion_2 = criterion_2.where(criterion_2 == 1, 0)

    # Combining
    criteria = criterion_1 * criterion_2
    ridges_labels = criteria.where(criteria == 1)
    return ridges_labels

for dt in range(ntimes):
    timeseq = np.arange(0, 8) + dt
    # lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, return_det=True)
    # det = lcs(ds.isel(time=timeseq))
    # det = np.sqrt(det)
    # potential_rainfall = (tcwv.isel(time=3) / (-6*3600*timeseq.shape[0])) * (1 - det)/det
    # potential_rainfall = - ( 2 * tcwv / (-6*3600*5) ) * (det - 1) / (det + 1)  # [mm / s]
    # potential_rainfall = potential_rainfall * 86400  # conversion to [mm / day]

    lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=2, subdomain=subdomain)
    ftle = lcs(ds.isel(time=timeseq))
    ftle_ = ftle.isel(time=0)
    # 14 for settls 4 and timeseq 4

    ridges = find_ridges_spherical_hessian(ftle_, sigma=1)
    ridges = ridges.where(ridges < -5e-9, 0)  # Warning: sensitive to sigma
    ridges = ridges.where(ridges >= -5e-9, 1)
    ftle_ = ftle_.interp(latitude=ridges.latitude, longitude=ridges.longitude)
    ridges = filter_ridges(ridges, ftle_, mean_intensity=25, min_area=5)


    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[16, 8])
    p = tcwv.sel(time=ftle.time.values).interp(latitude=ftle.latitude, longitude=ftle.longitude,
                                                method='linear').plot(ax=axs[0], cmap=cmr.gem, vmin=20, vmax=70,
                                                                      cbar_kwargs={'shrink': 0.8},
                                                                      transform=ccrs.PlateCarree())
    p.set_edgecolor('face')
    axs[0].coastlines(color='white')
    # axs[0].set_title('FTLE')

    uplot = u.sel(time=ftle.time.values[0]).interp(latitude=ftle.latitude, longitude= ftle.longitude, method='linear')
    vplot = v.sel(time=ftle.time.values[0]).interp(latitude=ftle.latitude, longitude= ftle.longitude, method='linear')
    mag = np.sqrt(uplot**2 + vplot**2).values
    axs[0].streamplot(x=uplot.longitude.values, y=vplot.latitude.values, u=uplot.values,
              v=vplot.values, color='white', transform=ccrs.PlateCarree(), linewidth=mag/6)
    ridges.plot(ax=axs[0], cmap=cmr.sunburst, alpha=.6, transform=ccrs.PlateCarree(), add_colorbar=False)
    p = ftle_.plot(ax=axs[1], cbar_kwargs={'shrink': 0.8}, vmin=0, vmax=100, cmap=cmr.rainforest,
                   transform=ccrs.PlateCarree())
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
