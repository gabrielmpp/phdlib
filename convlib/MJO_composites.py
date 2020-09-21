from dask.diagnostics import ProgressBar
import xarray_gab.xarray as xr
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import cmasher as cmr
import glob
import requests
from subprocess import call
import os
from xr_tools.tools import size_in_memory

def exists(URL):
    r = requests.head(URL)
    return r.status_code == requests.codes.ok


def download_and_sanitize(URL):
    columns_to_select = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude']
    assert exists(URL), 'Path does not exist'
    call(["curl", "-s", "-o", 'temp.txt', URL], stdout=open(os.devnull, 'wb'))
    df = pd.read_csv('temp.txt', skiprows=1, delim_whitespace=True)
    call(['rm', 'temp.txt'])
    new_columns = []
    for idx, column in enumerate(df.columns):
        new_columns.append(column.replace(',', '').replace('.', ''))
    df.columns = pd.Index(new_columns)
    df = df[columns_to_select]
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.drop(['year', 'month','day'], axis=1)
    df = df.set_index('time')
    ds = df.to_xarray()
    ds = ds.where(ds < 1e10)
    ds = ds.where(ds != 999)
    return ds

def read_u_v(years):
    u_list, v_list, z_list, pr_list, tcwv_list = [], [], [], [], []
    for year in np.arange(years.start, years.stop):
        z = xr.open_dataset(datapath + zfilename.format(year=year), chunks={'time': 100})
        u = xr.open_dataset(datapath + ufilename.format(year=year), chunks={'time': 100})
        v = xr.open_dataset(datapath + vfilename.format(year=year), chunks={'time': 100})
        pr = xr.open_dataset(datapath + prfilename.format(year=year), chunks={'time': 100})
        tcwv = xr.open_dataset(datapath + tcwvfilename.format(year=year), chunks={'time': 100})
        z = z.assign_coords(longitude=(z.coords['longitude'].values + 180) % 360 - 180)
        u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
        v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
        pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
        tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
        u = u.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        z = z.sel(latitude=slice(20, -70), longitude=slice(-160, -10))
        v = v.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        pr = pr.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        tcwv = tcwv.sel(latitude=slice(20, -60), longitude=slice(-80, -10))

        z = z.to_array().isel(variable=0).drop('variable')
        u = u.to_array().isel(variable=0).drop('variable')
        v = v.to_array().isel(variable=0).drop('variable')
        pr = pr.to_array().isel(variable=0).drop('variable')
        tcwv = tcwv.to_array().isel(variable=0).drop('variable')
        u_list.append(u)
        v_list.append(v)
        z_list.append(z)
        pr_list.append(pr)
        tcwv_list.append(tcwv)
        print(str(year))
    u = xr.concat(u_list, dim='time')
    v = xr.concat(v_list, dim='time')
    z = xr.concat(z_list, dim='time')
    pr = xr.concat(pr_list, dim='time')
    tcwv = xr.concat(tcwv_list, dim='time')
    [x.close() for x in u_list]
    [x.close() for x in v_list]
    [x.close() for x in z_list]
    [x.close() for x in pr_list]
    [x.close() for x in tcwv_list]
    u_list = None
    v_list = None
    z_list = None
    pr_list = None
    tcwv_list = None
    return u, v, z, pr, tcwv

# ---- Loading paths and contants ---- #
print('\n #---- Loading paths ----# \n')
outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
experiment = 'experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4/'
datapath = '/gws/nopw/j04/primavera1/observations/ERA5/'
vfilename = 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
ufilename = 'viwve_ERA5_6hr_{year}010100-{year}123118.nc'
zfilename = 'zg_250_ERA5_6hrPlevPt_{year}010100-{year}123118.nc'
prfilename = 'pr_ERA5_6hr_{year}010100-{year}123118.nc'
tcwvfilename = 'tcwv_ERA5_6hr_{year}010100-{year}123118.nc'
basins = ['Tiete', 'Uruguai']
seasons = [1, 3]

#
# URL = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'
#
# mjo_ds = download_and_sanitize(URL)
# mjo_ds = mjo_ds.resample(time='6H').nearest(tolerance='1D')
#
# # ---- Lazily read data ---- #
# print('\n #---- Reading data lazily ----# \n')
# years = slice(1981, 2010)
# mjo_ds = mjo_ds.sel(time=slice('1980', '2010'))
#
# u, v, z, pr, tcwv = read_u_v(years)
# with open(outpath + experiment + 'config.txt') as file:
#     config = eval(file.read())
# days = config['lcs_time_len'] / 4
# files = [f for f in glob.glob(outpath + experiment + "**/partial_ridges_0*.nc", recursive=True)]
# da = xr.open_mfdataset(files, preprocess=lambda x: x.sortby('time'))
# da = da.to_array()
# da = da.sortby('time')
# da = da.isel(variable=0).drop('variable')
# mjo_ds = mjo_ds.sel(time=da.time)
# da['phase'] = 'time', mjo_ds.phase.values
# da['amplitude'] = 'time', mjo_ds.amplitude.values
# da = da.where(da.amplitude > 1, drop=True)
# da = da.load()
# da = da.where(da == 1, 0)
# pr = pr.sel(time=da.time.values)
# pr['phase'] = 'time', da.phase.values
# pr['amplitude'] = 'time', da.amplitude.values
# pr = pr.load()
#
#
# da_djf = list(da.groupby('time.season'))[1][1]  # 0 summer, 1 winter in the first index
# pr_djf = list(pr.groupby('time.season'))[1][1]
# da_phase = da_djf.groupby('phase').mean('time')
# pr_phase = pr_djf.groupby('phase').mean('time')
# pr_phase = pr_phase.sel(latitude=da_phase.latitude,
#                         longitude=da_phase.longitude,
#                         method='nearest')
# da_phase = da_phase - da_phase.mean('phase')
# pr_phase = pr_phase - pr_phase.mean('phase')
# da_phase.name='% of CZ events in DJF per phase'
# p = (100*da_phase).plot(col='phase', col_wrap=4, vmax=3, vmin=-3, cmap='RdBu',
#                         subplot_kws={'projection': ccrs.PlateCarree()},
#               transform=ccrs.PlateCarree())
# for phase_idx, ax in enumerate(p.axes.flatten()):
#     ax.coastlines(color='black')
#
# pr_phase.name='Avg rainfall in DJF per phase (mm/day)'
# p = (86400*pr_phase/4).plot(col='phase', col_wrap=4, vmax=5, vmin=-5, cmap='RdBu',
#                         subplot_kws={'projection': ccrs.PlateCarree()},
#               transform=ccrs.PlateCarree())
# for phase_idx, ax in enumerate(p.axes.flatten()):
#     ax.coastlines(color='black')
#
#
#
# da_phase = da_phase - da_phase.mean('phase')
# size_in_memory(da)
#
# # ---- Laily calculate flux across Amazon borders ---- #
# print('\n #---- Laily calculate flux across Amazon borders ----# \n')
#
# dlat = np.sign(MAG['amazon'].diff('latitude'))
# dlon = np.sign(MAG['amazon'].diff('longitude'))
# border_amazon = np.abs(dlat) + np.abs(dlon)
# border_amazon = border_amazon.where(border_amazon <= 1, 1)
# influx = dlat * v + dlon * u
# influx = influx.where(border_amazon)
#
# # ---- Reading basin masks and calculating area ---- #
# print('\n #----Reading basin masks and calculating area ----# \n')
#
# assert pd.Index(da.time.values).duplicated().sum() == 0, 'There are repeated time indices'
# mask_tiete = MAG['Tiete'].interp(latitude=da.latitude, longitude=da.longitude, method='nearest')
# mask_uruguai = MAG['Uruguai'].interp(latitude=da.latitude, longitude=da.longitude, method='nearest')
# mask_tiete = mask_tiete.sel(latitude=slice(-30, -15), longitude=slice(-55, -30))
# mask_uruguai = mask_uruguai.sel(latitude=slice(-35, -15), longitude=slice(-65, -30))
# area_uruguai = mask_uruguai.sum()
# area_tiete = mask_tiete.sum()
#
# # ---- time sel and season ---- #
# pr = pr.sel(time=da.time)
# u = u.sel(time=da.time)
# v = v.sel(time=da.time)
# z = z.sel(time=da.time)
# influx = influx.sel(time=da.time)
#
#

# ----- Histograms

years = slice(1981, 2010)
with open(outpath + experiment + 'config.txt') as file:
    config = eval(file.read())
days = config['lcs_time_len'] / 4
files = [f for f in glob.glob(outpath + experiment + "**/partial_0*.nc", recursive=True)]
da = xr.open_mfdataset(files, preprocess=lambda x: x.sortby('time'))
da = da.to_array()
da = da.sortby('time')
da = da.isel(variable=0).drop('variable')

tcwv = read_u_v(years)[-1]
URL = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'

mjo_ds = download_and_sanitize(URL)
mjo_ds = mjo_ds.resample(time='6H').nearest(tolerance='1D')

# ---- Lazily read data ---- #
print('\n #---- Reading data lazily ----# \n')
years = slice(1981, 2010)
mjo_ds = mjo_ds.sel(time=slice('1980', '2010'))
mjo_ds = mjo_ds.sel(time=da.time)
da['phase'] = 'time', mjo_ds.phase.values
da['amplitude'] = 'time', mjo_ds.amplitude.values
da = da.where(da.amplitude > 1, drop=True)
da = np.log(da) / days
da = da.where(~xr.ufuncs.isinf(da), 0)
da = da.load()
da_djf = list(da.groupby('time.season'))[0][1]  # 0 summer, 1 winter in the first index



tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(time=da.time)
tcwv['amplitude'] = 'time', da.amplitude.values
tcwv['phase'] = 'time', da.phase.values
tcwv = tcwv.load()
tcwv_djf =  list(tcwv.groupby('time.season'))[0][1]

tcwv_se = tcwv_djf.sel(latitude=slice(-25, -15), longitude=slice(-50, -40))  # selecting

da_se = da.sel(latitude=slice(-25, -15), longitude=slice(-50, -40))  # selecting _SE
da_itcz = da.sel(latitude=slice(-10, None), longitude=slice(-50, None))  # selecting _SE


da_quants = da.quantile(dim=['latitude', 'longitude'], q=np.arange(0.1, 0.9, 0.1))
da_quants_se = da_se.quantile(dim=['latitude', 'longitude'], q=np.arange(0.05, 1, 0.1))
tcwv_quants_se = tcwv_se.quantile(dim=['latitude', 'longitude'], q=np.arange(0.05, 1, 0.1))
da_quants_itcz = da_itcz.quantile(dim=['latitude', 'longitude'], q=np.arange(0.05, 1, 0.1))
# da_quants_se = da_quants_se / np.abs(da_quants_se).max('time')
da_quants_se.name = 'FTLE (1/day)'
da_quants_itcz.name = 'FTLE (1/day)'
da_quants.name = 'FTLE (1/day)'

plt.style.use('bmh')

fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
width_tcwv = tcwv_quants_se.isel(quantile=-1) - tcwv_quants_se.isel(quantile=0)
width = da_quants_se.isel(quantile=-1) - da_quants_se.isel(quantile=0)
width.name = 'FTLE (q=0.95) - FTLE (q=0.05)'
width_tcwv.name = 'TCWV (q=0.95) - TCWV (q=0.05)'
width.groupby('phase').mean('time').plot(ax=ax, color='red')
width_tcwv.groupby('phase').mean('time').plot(ax=ax2, color='blue')
fig.legend(['FTLE width', 'TCWV width'])
ax2.grid(False)
plt.show()


fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
upper_tcwv = tcwv_quants_se.isel(quantile=-1)
upper = da_quants_se.isel(quantile=-1)
upper.name = 'FTLE (q=0.95) '
upper_tcwv.name = 'TCWV (q=0.95)'
upper.groupby('phase').mean('time').plot(ax=ax, color='red')
upper_tcwv.groupby('phase').mean('time').plot(ax=ax2, color='blue')
fig.legend(['FTLE 95%', 'TCWV 95%'])
ax2.grid(False)
ax.set_title(None)
ax2.set_title(None)
plt.show()

fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
lower_tcwv = tcwv_quants_se.isel(quantile=0)
lower = da_quants_se.isel(quantile=0)
lower.name = 'FTLE (q=0.05) '
lower_tcwv.name = 'TCWV (q=0.05)'
lower.groupby('phase').mean('time').plot(ax=ax, color='red')
lower_tcwv.groupby('phase').mean('time').plot(ax=ax2, color='blue')
fig.legend(['FTLE 5%', 'TCWV 5%'])
ax2.grid(False)
ax.set_title(None)
ax2.set_title(None)
plt.show()

da_quants_se_phase = da_quants_se.groupby('phase').mean('time')
tcwv_quants_se_phase = tcwv_quants_se.groupby('phase').mean('time')
da_quants_itcz_phase = da_quants_itcz.groupby('phase').mean('time')
da_quants_phase = da_quants.groupby('phase').mean('time')
import statsmodels.api as sm


sm.qqplot(da_quants_itcz_phase.sel(phase=1).values, line='s')
sm.qqplot(da_quants_itcz_phase.sel(phase=7).values, line='s')

plt.style.use('bmh')
fig, ax = plt.subplots(1, 1)
ax.scatter(da_quants_itcz_phase.sel(phase=7).values, da_quants_itcz_phase.sel(phase=1).values)
ax.plot([-0.25, 1.5], [-0.25, 1.5], color='black')
ax.legend(['1:1', 'Samples'])
ax.set_xlabel('Quantiles phase 7')
ax.set_ylabel('Quantiles phase 1')


fig, ax = plt.subplots(1, 1)
ax.scatter(da_quants_se_phase.sel(phase=4).values, da_quants_se_phase.sel(phase=1).values)
ax.plot([-0.25, 1.5], [-0.25, 1.5], color='black')
ax.legend(['1:1', 'Samples'])
ax.set_xlabel('Quantiles phase 4')
ax.set_ylabel('Quantiles phase 1')



fig, ax = plt.subplots(1, 1)
ax.scatter(tcwv_quants_se_phase.sel(phase=4).values, tcwv_quants_se_phase.sel(phase=1).values)
ax.plot([28, 50], [28, 50], color='black')
ax.legend(['1:1', 'Samples'])
ax.set_xlabel('Quantiles phase 4')
ax.set_ylabel('Quantiles phase 1')

da_quants_se.groupby('phase').mean('time').plot.scatter(x='quantile')

da.plot.hist()

plt.style.use('bmh')
fig, ax = plt.subplots(1, 1, sharey=True, sharex=True)
for idx in range(da.phase.values.shape[0]):
    da.where(da.phase==(idx+1), drop=True).plot.hist(ax=ax, bins=np.arange(-3, 5, .25), density=True)
    # ax.semilogy(True)
    ax.set_xlabel(f'FTLE phase {idx+1}')

for idx, ax in enumerate(axs.flatten()):
    ax.semilogy(False)
