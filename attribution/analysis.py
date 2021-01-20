import xarray as xr
import cmasher as cmr
import matplotlib.pyplot as plt
import glob
from dask.diagnostics import ProgressBar
import cartopy.crs as ccrs 
import numpy as np
import pandas as pd
import climIndices as ci
from mia_core.miacore.xrumap import AutoEncoder
from copy import deepcopy
from xr_tools.tools import read_nc_files, createDomains, size_in_memory


def calculate_anomaly(da, groupby_type="time.month"):
    gb = da.groupby(groupby_type)
    clim = gb.mean(dim="time", skipna=True)
    return clim - da.mean('time', skipna=True)


def calc_direction(angle):
    if angle <= -67.5 or angle > 67.5:
        return 1
    elif 22.5 < angle <= 67.5:
        return 2
    elif -22.5 < angle <= 22.5:
        return 3
    elif -67.5 < angle <= -22.5:
        return 4
    else:
        return 0


# ---- Paths ---- #
running_on = 'jasmin'
if running_on == 'nimbus':
    rainpath = '/media/gab/gab_hd2/data/rain/prec_monthly_UT_Brazil_v2_198001_201312.nc'
    # rainpath = '/media/gab/gab_hd2/data/rain/precip_1979a2017_CPC_AS.nc'
    ridgepath = '/home/gab/phd/data/ridges_smoothed_spline/'
    ftlepath = '/home/gab/phd/data/ftle_smoothed_spline/'
    anglepath = '/home/gab/phd/data/eigvectors_smoothed_spline/'
    da_rain = xr.open_dataset(rainpath)['prec']

elif running_on == 'jasmin':
    ERA5_path = '/gws/nopw/j04/primavera1/observations/ERA5/'
    experiment_path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/' \
                      'experiment_timelen_8_89d05342-6abb-4e0a-8c14-699dbdc78351/'



    rainpath = '/media/gab/gab_hd2/data/rain/prec_monthly_UT_Brazil_v2_198001_201312.nc'
    # rainpath = '/media/gab/gab_hd2/data/rain/precip_1979a2017_CPC_AS.nc'
    ridgepath = experiment_path + 'ridges/'
    ftlepath = experiment_path
    anglepath = experiment_path + 'angles/'

    da_rain = read_nc_files(region='SA', basepath=ERA5_path,
                  filename='pr_ERA5_6hr_{year}010100-{year}123118.nc', season=None, transformLon=True, reverseLat=True)
    da_tcwv = read_nc_files(region='SA', basepath=ERA5_path,
                  filename='tcwv_ERA5_6hr_{year}010100-{year}123118.nc', season=None, transformLon=True, reverseLat=True)

    da_rain = da_rain * 1000 #  m to mm
ridgefiles = glob.glob(ridgepath + '*.nc')
anglefiles = glob.glob(anglepath + '*.nc')
ftlefiles = glob.glob(ftlepath + '*.nc')


# nao = ci.get_data(['nao'])
# nao = nao.to_xarray()['nao'].rename(index='time')

ridge_list = []
for ridgefile in ridgefiles:
    ridge_list.append(xr.open_dataarray(ridgefile, chunks={'time':1}))

ftle_list = []
for ftlefile in ftlefiles:
    ftle_list.append(xr.open_dataarray(ftlefile, chunks={'time':1}))

angle_list = []
for anglefile in anglefiles:
    angle_list.append(xr.open_dataarray(anglefile, chunks={'time':1}))

da_angle = xr.concat(angle_list, dim='time')
ftle = xr.concat(ftle_list, dim='time')
ftle = ftle.chunk({'time': 600})
ridges = xr.concat(ridge_list, dim='time')
ridges = ridges.chunk({'time': 600})
# ridges = ridges.where(xr.ufuncs.isnan(ridges), 1)
ridges = ridges.where(~xr.ufuncs.isnan(ridges), 0)

# ---- RESULTS 1 - Equator ---- #
# Freq of occurrence and angle
equator = {'latitude': slice(-10, 10)}
nebr = {'latitude': [-10, 0],
        'longitude': [-42, -32]}
atlantic = {'latitude': [0, 10],
        'longitude': [-52, -32]}
with ProgressBar():
    occurrence_ridges = ridges.sel(**equator).groupby('time.season').mean('time').compute()
with ProgressBar():
    angle_mean = da_angle.sel(**equator).groupby('time.season').mean('time').compute()
occurrence_ridges = occurrence_ridges.coarsen(latitude=3, longitude=3, boundary='trim').sum()
angle_mean = angle_mean.coarsen(latitude=3, longitude=3, boundary='trim').mean()
plt.style.use('bmh')
titles = ['a) {}', 'b) {}', 'c) {}', 'd) {}', 'e) {}', 'f) {}', 'g) {}', 'h) {}']

fig, axs = plt.subplots(2, 4, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[10, 4])
for idx, season in enumerate(occurrence_ridges.season.values):
    ax = axs.T.flatten()[idx]

    rectangle1 = plt.Rectangle((nebr['longitude'][0], nebr['latitude'][0]), nebr['longitude'][1] - nebr['longitude'][0],
                               nebr['latitude'][1]-nebr['latitude'][0], fc=None, ec="red", fill=False, linewidth=2)
    rectangle2 = plt.Rectangle((atlantic['longitude'][0], atlantic['latitude'][0]), atlantic['longitude'][1] - atlantic['longitude'][0],
                               atlantic['latitude'][1]-atlantic['latitude'][0], fc=None, ec="red", fill=False,linewidth=2)
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    p1 = (100*occurrence_ridges).sel(season=season).plot(vmax=80, cmap='viridis', transform=ccrs.PlateCarree(), ax=ax,
                                                  add_colorbar=False)
    occurrence_ridges.sel(season=season).plot.contour(vmax=0.8, colors='k', transform=ccrs.PlateCarree(), ax=ax,
                                              add_colorbar=False, linewidths=.2, levels=np.arange(0.2, 1, .2))
    ax.coastlines()
    ax.set_title('')
    ax.set_title(titles[idx].format(season),  loc='left')
    ax = axs.T.flatten()[idx + 4]
    p2 = angle_mean.sel(season=season).plot(vmax=50, vmin=-50,
                                                transform=ccrs.PlateCarree(), ax=ax, add_colorbar=False,
                                                cmap=cmr.pride)
    angle_mean.sel(season=season).plot.contour(levels=[-30, -15,  15, 30],vmax=50, vmin=-50, linewidths=.2,
                                               transform=ccrs.PlateCarree(), ax=ax, add_colorbar=False,
                                            colors=['k'], )
    ax.set_title('')
    ax.set_title(titles[idx+4].format(season),  loc='left')

    ax.coastlines()
cbar2 = plt.colorbar(p2, ax=axs[:, [2, 3]], orientation='horizontal', ticks=[-30, -15, 0 , 15, 30])
cbar1 = plt.colorbar(p1, ax=axs[:, [0, 1]], orientation='horizontal')
cbar2.ax.set_xlabel('LCSs average orientation (deg)')
cbar1.ax.set_xlabel('LCSs freq. of occurrence (%)')
plt.savefig('figs/occurence_equator.png', dpi=600)
plt.close()

#  Occurrence and angle time series

da_angle_nebr = da_angle.sel(latitude=slice(*nebr['latitude']), longitude=slice(*nebr['longitude']))
da_angle_atlantic = da_angle.sel(latitude=slice(*atlantic['latitude']), longitude=slice(*atlantic['longitude']))
da_angle_nebr_ts = da_angle_nebr.mean(['latitude', 'longitude'], skipna=True)
da_angle_atlantic_ts = da_angle_atlantic.mean(['latitude', 'longitude'], skipna=True)
with ProgressBar():
    da_angle_nebr_ts = da_angle_nebr_ts.compute(num_workers=8)
with ProgressBar():
    da_angle_atlantic_ts = da_angle_atlantic_ts.compute(num_workers=8)
da_angle_atlantic_ts.sortby('time').sel(time=slice('2004','2008')).plot()
da_angle_nebr_ts.sortby('time').sel(time=slice('2004','2008')).plot()
# ridges['angle'] = 'time', da_angle_ts.values
da_occurrence_nebr_ts = da_angle_nebr_ts.where(xr.ufuncs.isnan(da_angle_nebr_ts), 1)
da_occurrence_nebr_ts = da_occurrence_nebr_ts.where(~xr.ufuncs.isnan(da_occurrence_nebr_ts), 0)
da_occurrence_monthly_nebr = da_occurrence_nebr_ts.groupby('time.month').mean('time') * 100
da_occurrence_atlantic_ts = da_angle_atlantic_ts.where(xr.ufuncs.isnan(da_angle_atlantic_ts), 1)
da_occurrence_atlantic_ts = da_occurrence_atlantic_ts.where(~xr.ufuncs.isnan(da_occurrence_atlantic_ts), 0)
da_occurrence_monthly_atlantic = da_occurrence_atlantic_ts.groupby('time.month').mean('time') * 100

# da_angle_daily = da_angle_ts.sortby('time').resample(time='1D').mean('time', skipna=True)
da_rain = da_rain.resample(time='1D').sum('time', skipna=True)
titles = [r'a) $\theta > 0$ Rain', r'b) $\theta < 0$ Rain', 'c) No LCS - Rain',
          r'd) $\theta > 0$ TCWV', r'e) $\theta < 0$ TCWV', 'f) No LCS - TCWV']
da_rain_1_nebr = da_rain.where(da_angle_nebr_ts.sortby('time').resample(time='1D').mean(skipna=True) >= 0, drop=True)
da_rain_2_nebr = da_rain.where(da_angle_nebr_ts.sortby('time').resample(time='1D').mean(skipna=True) < 0, drop=True)
da_tcwv_1_nebr = da_tcwv.where(da_angle_nebr_ts.sortby('time').resample(time='1D').mean(skipna=True) >= 0, drop=True)
da_tcwv_2_nebr = da_tcwv.where(da_angle_nebr_ts.sortby('time').resample(time='1D').mean(skipna=True) < 0, drop=True)
da_rain_3_nebr = da_rain.where(np.isnan(da_angle_nebr_ts.sortby('time').resample(time='1D').mean(skipna=True)), drop=True)
da_tcwv_3_nebr = da_tcwv.where(np.isnan(da_angle_nebr_ts.sortby('time').resample(time='1D').mean(skipna=True)), drop=True)

with ProgressBar():
    ridges_mean_1_nebr = ridges.where(da_angle_nebr_ts >= 0).sum('time').compute(num_workers=8)
with ProgressBar():
    ridges_mean_2_nebr = ridges.where(da_angle_nebr_ts < 0).sum('time').compute(num_workers=8)


fig, axs = plt.subplots(2, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=[8, 5])
p = da_rain_1_nebr.mean('time').plot.contourf(levels=11,ax=axs[0, 0], vmin=0, vmax=4, add_colorbar=False, cmap=cmr.freeze_r)
da_rain_2_nebr.mean('time').plot.contourf(levels=11,ax=axs[0, 1], vmin=0, vmax=4, add_colorbar=False, cmap=cmr.freeze_r)
da_rain_3_nebr.mean('time').plot.contourf(levels=11,ax=axs[0, 2], vmin=0, vmax=4, add_colorbar=False, cmap=cmr.freeze_r)
p2 = (da_tcwv_1_nebr - da_tcwv.mean('time')).mean('time').plot.contourf(levels=21,ax=axs[1, 0], vmin=-10, vmax=10,  add_colorbar=False, cmap=cmr.fusion)
(da_tcwv_2_nebr - da_tcwv.mean('time')).mean('time').plot.contourf(levels=21,ax=axs[1, 1], vmin=-10, vmax=10, add_colorbar=False, cmap=cmr.fusion)
(da_tcwv_3_nebr - da_tcwv.mean('time')).mean('time').plot.contourf(levels=21,ax=axs[1, 2], vmin=-10, vmax=10, add_colorbar=False, cmap=cmr.fusion)
ridges_mean_1_nebr.sel(latitude=slice(*nebr['latitude']), longitude=slice(*nebr['longitude'])).plot.contour(levels=5, ax=axs[0, 0], colors='red', linewidths=.5)
ridges_mean_2_nebr.sel(latitude=slice(*nebr['latitude']), longitude=slice(*nebr['longitude'])).plot.contour(levels=5, ax=axs[0, 1], colors='red', linewidths=.5)
ridges_mean_1_nebr.sel(latitude=slice(*nebr['latitude']), longitude=slice(*nebr['longitude'])).plot.contour(levels=5, ax=axs[1, 0], colors='red', linewidths=.5)
ridges_mean_2_nebr.sel(latitude=slice(*nebr['latitude']), longitude=slice(*nebr['longitude'])).plot.contour(levels=5, ax=axs[1, 1], colors='red', linewidths=.5)
cbar1 = plt.colorbar(p, ax=axs[0,:], orientation='vertical', shrink=.8)
cbar2 = plt.colorbar(p2, ax=axs[1,:], orientation='vertical', shrink=.8)
cbar1.ax.set_ylabel(r'Precip. $\frac{mm}{day}$')
cbar2.ax.set_ylabel(r'TCWV anom. $\frac{kg}{m^3}$')
for idx, ax in enumerate(axs.flatten()):
    rectangle = plt.Rectangle((nebr['longitude'][0], nebr['latitude'][0]), nebr['longitude'][1] - nebr['longitude'][0],
                               nebr['latitude'][1] - nebr['latitude'][0], fc=None, ec="red", fill=False, linewidth=2)
    ax.coastlines()
    ax.set_title(titles[idx])
    ax.add_patch(rectangle)
    ax.set_xlim([None, -30])
plt.savefig('figs/rain_tcwv_nebr.png', dpi=600)
plt.close()



da_rain_2_nebr = da_rain.where((da_angle_nebr_ts > 0) & (da_angle_nebr_ts < 15))
da_rain_3_nebr = da_rain.where((da_angle_nebr_ts <= 0) & (da_angle_nebr_ts > -15))
da_rain_4_nebr = da_rain.where(da_angle_nebr_ts <= -15)
da_rain_5_nebr = da_rain.where(xr.ufuncs.isnan(da_angle_nebr_ts))

with ProgressBar():
    ridges_mean_1 = ridges.where(ridges.angle > 15).sum('time').compute(num_workers=4)
ridges_mean_1 = ridges_mean_1.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
ridges_mean_1.plot()
plt.show()
with ProgressBar():
    ridges_mean_2 = ridges.where((ridges.angle > 0) &
                                                (ridges.angle < 15)).sum('time').compute(num_workers=4)

ridges_mean_2 = ridges_mean_2.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
with ProgressBar():
    ridges_mean_3 = ridges.where((ridges.angle >= -15) &
                                                (ridges.angle < 0)).sum('time').compute(num_workers=4)
ridges_mean_3 = ridges_mean_2.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
with ProgressBar():
    ridges_mean_4= ridges.where(ridges.angle <= -15).sum('time').compute(num_workers=4)

ridges_mean_4 = ridges_mean_4.sel(latitude=slice(*region_boundaries['latitude']),






region = 'sallj'
regions = dict(
nebr = {'latitude': [-8, 2],
        'longitude': [-42, -32]},
sallj = {'latitude': [-22, -12],
        'longitude': [-70, -60]}
)
region_boundaries = regions[region]

# --- Rain PCA --- #

xru_model = AutoEncoder(dims_to_reduce=['longitude', 'latitude'], alongwith=['time'], mode='pca')
xru_model = xru_model.fit(da_rain)
scores = xru_model.transform(da_rain)
scores = scores.rename(encoded_dims='PC')
scores = scores.assign_coords(PC=scores.PC.values + 1).isel(PC=slice(0, 6))
loadings = deepcopy(xru_model.reducer.components_)
loadings_coords = deepcopy(xru_model.stacked_coords)
loadings = xr.DataArray(loadings, dims=['alongwith', 'dim_to_reduce'], coords=loadings_coords)
loadings = loadings.unstack().rename(time='PC')
loadings = loadings.assign_coords(PC=np.arange(loadings.PC.values.shape[0])+1).sortby('latitude').sortby('longitude').\
    transpose('latitude', 'longitude', ...).isel(PC=slice(0, 6))
loadings.plot(col='PC',cmap=cmr.fusion, col_wrap=3)
plt.savefig('PCAs.png', dpi=600,  pad_inches=.2, bbox_inches='tight')
plt.close()
plt.style.use('ggplot')
scores.sel(time=slice('2008', '2012')).plot.line(x='time')
plt.savefig('PCAs_scores_.png', dpi=600,  pad_inches=.2, bbox_inches='tight')
plt.close()
# --- Plot region ---- #
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
rectangle = plt.Rectangle((region_boundaries['longitude'][0], region_boundaries['latitude'][0]), 10, 10, fc='blue',ec="red")
plt.gca().add_patch(rectangle)
da_angle.sortby('time').isel(time=10).plot(ax=ax)
ax.coastlines()
plt.show()
ridges = ridges.chunk({'time': 600, 'latitude': 30, 'longitude': 30})

ridges_region = ridges.sel(latitude=slice(*region_boundaries['latitude']),
                       longitude=slice(*region_boundaries['longitude']))
# from xr_tools.tools import size_in_memory
# size_in_memory(ridges)
# with ProgressBar():
#     ftle_region = ridges_region.compute(num_workers=4)

da_angle_nebr = da_angle.sel(latitude=slice(*region_boundaries['latitude']), longitude=slice(*region_boundaries['longitude']))
da_angle_ts = da_angle_nebr.mean(['latitude', 'longitude'], skipna=True)
with ProgressBar():
    da_angle_ts = da_angle_ts.compute(num_workers=5)
ridges['angle'] = 'time', da_angle_ts.values
da_occurrence_ts = da_angle_ts.where(xr.ufuncs.isnan(da_angle_ts), 1)
da_occurrence_ts = da_occurrence_ts.where(~xr.ufuncs.isnan(da_occurrence_ts), 0)
da_occurrence_monthly = da_occurrence_ts.groupby('time.month').mean('time') * 100

# --- Plotting PC scores and angle --- #
PC_n = 3
fig, ax = plt.subplots(1, 1, figsize=[12, 6])
ax2 = ax.twinx()
da_angle_ts.sortby('time').sel(time=slice('1993', '1998')).plot(ax=ax, color='blue')
da_angle_ts.sortby('time').sel(time=slice('1993', '1998')).resample(time='1M').mean('time').plot(color='k', ax=ax, linewidth=2)
scores.sel(PC=PC_n, time=slice('1993', '1998')).plot(ax=ax2, color='red')
ax.set_ylabel('Angles', color='blue')
ax2.set_ylabel(f'PC{PC_n} scores', color='red')
plt.savefig(f'angles_pc{PC_n}_scores_{region}.png', dpi=600,   pad_inches=.2, bbox_inches='tight')
plt.close()

xr.corr()
# --- Ridhes features --- #
from xr_tools.tools import size_in_memory, filter_ridges
from skimage.measure import regionprops_table
from scipy.ndimage import label


size_in_memory(ridges) # too big cant load

df = pd.DataFrame(regionprops_table(ridges_labels.values,
                                    intensity_image=ftle.values,
                                    properties=props))

monthly_ts = da_angle_ts.groupby('time.month').mean('time')
monthly_stdev = np.sqrt(da_angle_ts.groupby('time.month').var('time'))
monthly_ts.plot()
plt.errorbar(monthly_ts.month.values, monthly_ts.values, yerr = .5*monthly_stdev.values)
plt.ylabel('Angle')
plt.show()

da_angle_ts_month = da_angle_ts.sortby('time').resample(time='1MS').mean('time')
angle_nao = xr.concat([da_angle_ts_month, nao], dim=pd.Index(['angle', 'nao'], name='Var'))
angle_nao = angle_nao.dropna('time', how='any')
angle_nao = angle_nao.groupby('time.year').mean('time')
# angle_nao['month'] = ('time', [int(pd.Timestamp(x).strftime('%m')) for x in angle_nao.time.values])
# angle_nao['year'] = ('time', [int(pd.Timestamp(x).strftime('%Y')) for x in angle_nao.time.values])
# angle_nao = angle_nao.where(angle_nao.month == 1, drop=True)

plt.scatter(angle_nao.sel(Var='angle').values,
            angle_nao.sel(Var='nao').values,)
for idx in range(angle_nao.year.values.shape[0]):
    plt.text(x=angle_nao.sel(Var='angle').isel(year=idx).values,
             y=angle_nao.sel(Var='nao').isel(year=idx).values,
             s=angle_nao.year.isel(year=idx).values)
plt.xlabel('Angle')
plt.ylabel('NAO')
plt.show()
plt.close()

angle_nao = xr.concat([da_angle_ts_month, nao], dim=pd.Index(['angle', 'nao'], name='Var'))
angle_nao = angle_nao.dropna('time', how='any')
angle_nao = angle_nao.groupby('time.year').mean('time')
# angle_nao = angle_nao.resample(time='1M').mean('time')
# angle_nao['month'] = ('time', [int(pd.Timestamp(x).strftime('%m')) for x in angle_nao.year.values])
# angle_nao = angle_nao.where(angle_nao.month == 11, drop=True)
# angle_nao['year'] = ('time', [int(pd.Timestamp(x).strftime('%Y')) for x in angle_nao.time.values])
# angle_nao = angle_nao.where(angle_nao.month == 1, drop=True)

plt.scatter(angle_nao.sel(Var='angle').values,
            angle_nao.sel(Var='nao').values,)
for idx in range(angle_nao.year.values.shape[0]):
    plt.text(x=angle_nao.sel(Var='angle').isel(year=idx).values,
             y=angle_nao.sel(Var='nao').isel(year=idx).values,
             s=angle_nao.year.isel(year=idx).values)
plt.xlabel('Angle')
plt.ylabel('NAO')
plt.show()
plt.close()



da_occurrence_monthly.plot()
plt.ylabel('Frequency of occurrence (%)')
plt.show()

da_angle_daily = da_angle_ts.sortby('time').resample(time='1D').mean('time', skipna=True)
titles = [r'$\theta > 25$', r'$0 < \theta < 25$', r'$-25 < \theta < 0$', r'$\theta < -25$']
da_rain_1 = da_rain.where(da_angle_daily >= 15)
da_rain_2 = da_rain.where((da_angle_daily > 0) & (da_angle_daily < 15))
da_rain_3 = da_rain.where((da_angle_daily <= 0) & (da_angle_daily > -15))
da_rain_4 = da_rain.where(da_angle_daily <= -15)

with ProgressBar():
    ridges_mean_1 = ridges.where(ridges.angle > 15).sum('time').compute(num_workers=4)
ridges_mean_1 = ridges_mean_1.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
ridges_mean_1.plot()
plt.show()
with ProgressBar():
    ridges_mean_2 = ridges.where((ridges.angle > 0) &
                                                (ridges.angle < 15)).sum('time').compute(num_workers=4)

ridges_mean_2 = ridges_mean_2.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
with ProgressBar():
    ridges_mean_3 = ridges.where((ridges.angle >= -15) &
                                                (ridges.angle < 0)).sum('time').compute(num_workers=4)
ridges_mean_3 = ridges_mean_2.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
with ProgressBar():
    ridges_mean_4= ridges.where(ridges.angle <= -15).sum('time').compute(num_workers=4)

ridges_mean_4 = ridges_mean_4.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))

fig, axs = plt.subplots(1, 4, subplot_kw={'projection': ccrs.PlateCarree()})
p = da_rain_1.mean('time').plot.contourf(levels=11,ax=axs[0], vmin=0, vmax=12, add_colorbar=False)
da_rain_2.mean('time').plot.contourf(levels=11,ax=axs[1], vmin=0, vmax=12, add_colorbar=False)
da_rain_3.mean('time').plot.contourf(levels=11,ax=axs[2], vmin=0, vmax=12, add_colorbar=False)
da_rain_4.mean('time').plot.contourf(levels=11,ax=axs[3], vmin=0, vmax=12, add_colorbar=False)
ridges_mean_1.plot.contour(levels=5, ax=axs[0], colors='k', linewidths=.2)
ridges_mean_2.plot.contour(levels=5, ax=axs[1], colors='k', linewidths=.2)
ridges_mean_3.plot.contour(levels=5, ax=axs[2], colors='k', linewidths=.2)
ridges_mean_4.plot.contour(levels=5, ax=axs[3], colors='k', linewidths=.2)
plt.colorbar(p, ax=axs, orientation='horizontal', shrink=.8)
for idx, ax in enumerate(axs):
    rectangle = plt.Rectangle((region_boundaries['longitude'][0], region_boundaries['latitude'][0]), 10, 10, fill=False, ec="red")

    ax.coastlines()
    ax.set_title(titles[idx])
    ax.add_patch(rectangle)
    ax.set_xlim([None, -30])
plt.savefig(f'precip_angle_{region}.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()

month = 'January'
if month == 'January' and region == 'nebr':
    groupby_idx = 0
    angle_lims = [20, [10, 20], [0, 10], 0]  # open ended
    vlims = [0, 12]
if month == 'July' and region == 'nebr':
    groupby_idx = 6
    angle_lims = [-7.5, [-10, -7.5], [-12.5, -10], -12.5]  # open ended
    vlims = [0, 12]
if month == 'January' and region == 'sallj':
    groupby_idx = 0
    angle_lims = [0, [-15, 0], [-30, -15], -30]  # open ended

    vlims = [0, 12]
if month == 'July' and region == 'sallj':
    groupby_idx = 6
    angle_lims = [-35, [-37.5, -35], [-40, -35], -35]  # open ended
    vlims = [0, 12]
da_angle_month_year = list(da_angle_daily.groupby('time.month'))[groupby_idx][1]
da_rain_month_year = list(da_rain.groupby('time.month'))[groupby_idx][1]
da_angle_month_year = da_angle_month_year.groupby('time.year').mean('time')
da_rain_month_year = da_rain_month_year.groupby('time.year').mean('time')

with ProgressBar():
    ridges_mean_1 = ridges.where(ridges.angle > angle_lims[0]).sum('time').compute(num_workers=4)
ridges_mean_1 = ridges_mean_1.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
ridges_mean_1.plot()
plt.show()
with ProgressBar():
    ridges_mean_2 = ridges.where((ridges.angle > angle_lims[1][0]) &
                                                (ridges.angle < angle_lims[1][1])).sum('time').compute(num_workers=4)
ridges_mean_2 = ridges_mean_2.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
with ProgressBar():
    ridges_mean_3 = ridges.where((ridges.angle >= angle_lims[2][0]) &
                                                (ridges.angle < angle_lims[2][1])).sum('time').compute(num_workers=4)
ridges_mean_3 = ridges_mean_2.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))
with ProgressBar():
    ridges_mean_4= ridges.where(ridges.angle <= angle_lims[3]).sum('time').compute(num_workers=4)

ridges_mean_4 = ridges_mean_4.sel(latitude=slice(*region_boundaries['latitude']),
                                 longitude=slice(*region_boundaries['longitude']))

num_of_years_1 = (da_angle_month_year > angle_lims[0]).sum().values
num_of_years_2 = ((da_angle_month_year > angle_lims[1][0]) &
                                                (da_angle_month_year < angle_lims[1][1])).sum().values
num_of_years_3 = ((da_angle_month_year >= angle_lims[2][0]) &
                                                (da_angle_month_year < angle_lims[2][1])).sum().values
num_of_years_4 = (da_angle_month_year <= angle_lims[3]).sum().values
num_of_years = [num_of_years_1, num_of_years_2, num_of_years_3, num_of_years_4]
da_rain_month_year_1 = da_rain_month_year.where(da_angle_month_year > angle_lims[0])
da_rain_month_year_2 = da_rain_month_year.where((da_angle_month_year > angle_lims[1][0]) &
                                                (da_angle_month_year < angle_lims[1][1]))
da_rain_month_year_3 = da_rain_month_year.where((da_angle_month_year >= angle_lims[2][0]) &
                                                (da_angle_month_year < angle_lims[2][1]))
da_rain_month_year_4 = da_rain_month_year.where(da_angle_month_year <= angle_lims[3])



plt.plot(da_angle_month_year.year.values, da_angle_month_year.values)
plt.xlim([da_angle_month_year.year.values[0], da_angle_month_year.year.values[-1]])
plt.title(rf'Mean $\theta$ on {month}')
plt.ylabel('Angle')
plt.show()

titles = [rf'$\theta > {angle_lims[0]}$', rf'${angle_lims[1][0]} < \theta < {angle_lims[1][1]}$',
          rf'${angle_lims[2][0]} < \theta < {angle_lims[2][1]}$', rf'$\theta < {angle_lims[3]}$']



fig, axs = plt.subplots(1, 4, subplot_kw={'projection': ccrs.PlateCarree()})
p = da_rain_month_year_1.mean('year').plot.contourf(levels=21,ax=axs[0], vmin=vlims[0], vmax=vlims[1], add_colorbar=False)
da_rain_month_year_2.mean('year').plot.contourf(levels=21,ax=axs[1], vmin=vlims[0], vmax=vlims[1], add_colorbar=False)
da_rain_month_year_3.mean('year').plot.contourf(levels=21,ax=axs[2], vmin=vlims[0], vmax=vlims[1], add_colorbar=False)
da_rain_month_year_4.mean('year').plot.contourf(levels=21,ax=axs[3], vmin=vlims[0], vmax=vlims[1], add_colorbar=False)
ridges_mean_1.plot.contour(levels=5, ax=axs[0], colors='k', linewidths=.2)
ridges_mean_2.plot.contour(levels=5, ax=axs[1], colors='k', linewidths=.2)
ridges_mean_3.plot.contour(levels=5, ax=axs[2], colors='k', linewidths=.2)
ridges_mean_4.plot.contour(levels=5, ax=axs[3], colors='k', linewidths=.2)
plt.colorbar(p, ax=axs, orientation='horizontal', shrink=.8)
for idx, ax in enumerate(axs):
    rectangle = plt.Rectangle((region_boundaries['longitude'][0], region_boundaries['latitude'][0]), 10, 10, fill=False, ec="red")
    ax.text(x=-35, y=-30, s=str(num_of_years[idx]))
    ax.coastlines()
    ax.set_title(titles[idx])
    ax.add_patch(rectangle)
    ax.set_xlim([None, -30])
plt.savefig(f'precip_angle_year_{month}_{region}.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()

# --- Calculation --- #

with ProgressBar():
    counts = ridges.sum('time').compute(num_workers=5)
# template = xr.concat([counts for i in np.arange(1, 13)], dim=pd.Index(np.arange(1, 13), name='month'))

with ProgressBar():
    monthly_mean_hog = da_hog.groupby('time.month').mean('time').compute(num_workers=5)

with ProgressBar():
    yearly_mean_hog = da_hog.groupby('time.year').mean('time').compute(num_workers=5)

with ProgressBar():
    monthly_mean = ridges.groupby('time.month').sum('time').compute(num_workers=5)

with ProgressBar():
    yearly_mean = ridges.groupby('time.year').sum('time').compute(num_workers=5)

with ProgressBar():
    season_mean_angle = da_angle.groupby('time.season').mean('time').compute(num_workers=5)

with ProgressBar():
    season_var_angle = da_angle.groupby('time.season').var('time').compute(num_workers=5)

season_stdev_angle = np.sqrt(season_var_angle)
n = da_angle.time.shape[0] / 4
t_test_angle = (n ** 0.5) * \
           season_mean_angle / season_stdev_angle
t_threshold = 2.807    # 99% bicaudal test

p = season_mean_angle.plot(
    col='season', col_wrap=2, subplot_kws={'projection': ccrs.PlateCarree()}, cmap=cmr.pride)
for idx, ax in enumerate(p.axes.flatten()):
    np.abs(t_test_angle).isel(season=idx).plot.contourf(ax=ax, hatches=['......', '   '], linewidths=.8,
                                 levels=[0, t_threshold], alpha=0, add_colorbar=False)
    ax.coastlines()
plt.savefig('angles.png', dpi=600)
plt.close()



# stacked = season_mean_angle.stack(points=['season', 'latitude', 'longitude'])
# direction = xr.apply_ufunc(calc_direction, stacked.groupby('points'))
# direction = direction.unstack()
# p = direction.plot(
#     col='season', col_wrap=2, subplot_kws={'projection': ccrs.PlateCarree()}, cmap='tab10',levels=[0, 1, 2, 3, 4, 5])
# for idx, ax in enumerate(p.axes.flatten()):
#
#     ax.coastlines()
# plt.savefig('directions.png', dpi=600)
# plt.close()


scaling_year = 4 * 365.4
anomaly_year = (yearly_mean/scaling_year) - counts
anomaly_year = 4 * 365.4 * anomaly_year
anomaly_year.isel(year=0).plot()
plt.show()
# monthly_mean = monthly_mean.coarsen(latitude=4, longitude=4, boundary='trim').mean()
scaling = 4 * 29 * xr.DataArray([31,28.4,31,30,31,30,31,31,30,31,30,31], dims='month', coords={'month':monthly_mean.month.values})

# counts = counts.coarsen(latitude=4, longitude=4, boundary='trim').mean()
anomaly = (monthly_mean / scaling) - counts
anomaly = 4 * anomaly *  xr.DataArray([31,28.4,31,30,31,30,31,31,30,31,30,31], dims='month', coords={'month':monthly_mean.month.values})

p = anomaly.plot(col='month', col_wrap=4, subplot_kws={'projection': ccrs.PlateCarree()})
for ax in p.axes.flatten():
    ax.coastlines()
plt.show()

rain_anomaly = da_rain.groupby('time.month').mean('time') - da_rain.mean('time')
yearly_rain_anomaly = da_rain.groupby('time.year').mean('time') - da_rain.mean('time')
rain_mean = da_rain.groupby('time.month').mean('time')
plt.style.use('default')

# --- Plots --- #

p = rain_anomaly.plot(figsize=[16, 12],col='month', col_wrap=4,cmap=cmr.fusion,robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()}, add_colorbar=False)
p.add_colorbar(shrink=.8)
for idx, ax in enumerate(p.axes.flatten()):
    c = anomaly.isel(month=idx).plot.contour(colors='gray', levels=[-12, -8, -4, 4, 8, 12], ax=ax,
                                             linestyles=['--','--', '--', '-', '-', '-'])
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines(color='blue')
    ax.set_ylim([-35, anomaly.latitude.max().values])
    ax.set_xlim([-70, anomaly.longitude.max().values])
plt.savefig('anomalies.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()

yearly_rain_anomaly = yearly_rain_anomaly.sel(year=anomaly_year.year)
p = yearly_rain_anomaly.plot(figsize=[18, 12],col='year', col_wrap=8,cmap=cmr.fusion,robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()}, add_colorbar=False)
p.add_colorbar(shrink=.8)
for idx, ax in enumerate(p.axes.flatten()):
    if idx == anomaly_year.year.shape[0]:
        break
    c = anomaly_year.sel(latitude=slice(-30, None)).isel(year=idx).plot.contour(colors='gray', levels=[-120, -80, -40, 40, 80, 120], ax=ax,
                                             linestyles=['--','--', '--', '-', '-', '-'])
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines(color='blue')
    ax.set_ylim([-35, anomaly.latitude.max().values])
    ax.set_xlim([-70, anomaly.longitude.max().values])
plt.savefig('anomalies_year.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()

yearly_anomaly_hog = yearly_mean_hog - yearly_mean_hog.mean('year')
p = yearly_anomaly_hog.plot(figsize=[18, 12], col='year', col_wrap=8,cmap='RdBu',robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()}, add_colorbar=False)
p.add_colorbar(shrink=.8)
for idx, ax in enumerate(p.axes.flatten()):
    if idx == anomaly_year.year.shape[0]:
        break
    c = yearly_rain_anomaly.isel(year=idx).plot.contour(levels=[-50, 0, 50], ax=ax,
                                                             linewidth=.1, alpha=.5, colors='k',
                                                        linestyles=['dotted', 'dashed', 'solid'])
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines()
    ax.set_ylim([-35, anomaly.latitude.max().values])
    ax.set_xlim([-70, anomaly.longitude.max().values])

plt.savefig('anomalies_year_hog.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()


p = rain_mean.plot(figsize=[16, 12], col='month', col_wrap=4,cmap='viridis',robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()}, add_colorbar=False)
p.add_colorbar(shrink=.8)

for idx, ax in enumerate(p.axes.flatten()):
    c = (monthly_mean/29).isel(month=idx).plot.contour(cmap=cmr.gem, levels=[ 1,2,4,8,16, 32, 64, 128], ax=ax)
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines(color='blue')
    ax.set_ylim([-35, anomaly.latitude.max().values])
    ax.set_xlim([-70, anomaly.longitude.max().values])
plt.savefig('monthly_mean.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()



nbr = dict(latitude=slice(-30, 10), longitude=slice(-70,None))
nebr = dict(latitude=slice(-10, 5), longitude=slice(-65,None))


p = rain_anomaly.sel(**nebr).plot(col='month', col_wrap=4,cmap=cmr.pride,robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()})
for idx, ax in enumerate(p.axes.flatten()):
    c = anomaly.isel(month=idx).sel(**nebr).plot.contour(levels=[-15, -10, -5, 5, 10, 15], ax=ax)
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines()
plt.savefig('anomalies_nebr.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()

ftle = ftle.transpose(..., 'time')


# monthly_mean_hog = monthly_mean_hog * monthly_mean.interp(latitude=da_hog.latitude, longitude=da_hog.longitude,
#         method='nearest') / scaling
# monthly_mean_hog = monthly_mean_hog / ftle.time.shape[0]

monthly_anomaly_hog = monthly_mean_hog - monthly_mean_hog.mean('month')


p = monthly_anomaly_hog.sel(**nbr).plot(figsize=[18, 12], col='month', col_wrap=4,cmap='RdBu',robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()}, add_colorbar=False)
p.add_colorbar(shrink=.7)
for idx, ax in enumerate(p.axes.flatten()):
    c = rain_anomaly.sel(**nbr).isel(month=idx).plot.contour(levels=[-150, -100, -50, 50, 100, 150], ax=ax,
                                                             linewidth=.1, alpha=.5, cmap='RdBu')
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines()
    ax.set_ylim([-35, anomaly.latitude.max().values])
    ax.set_xlim([-70, anomaly.longitude.max().values])
plt.savefig('anomalies_hog_nbr.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()


p = monthly_mean_hog.sel(**nbr).plot(figsize=[18, 12], col='month', col_wrap=4,cmap='Blues',robust=True,
        subplot_kws={'projection': ccrs.PlateCarree()}, add_colorbar=False)
p.add_colorbar(shrink=.7)

for idx, ax in enumerate(p.axes.flatten()):
    c = rain_mean.sel(**nbr).isel(month=idx).plot.contour(levels=[50, 100, 150, 200, 250], ax=ax,
                                                          linewidth=.1, alpha=.5, cmap='viridis')
    plt.clabel(c, fmt='%1.0f')
    ax.coastlines()
plt.savefig('mean_hog_nbr.png', dpi=600, pad_inches=.2, bbox_inches='tight')
plt.close()


