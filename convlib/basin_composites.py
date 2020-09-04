"""
Script to calculate means and anomalies during CZ events per basin
"""
from dask.diagnostics import ProgressBar
import xarray_gab.xarray as xr
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import cmasher as cmr
import glob

def read_u_v(years):
    u_list, v_list, z_list, pr_list = [], [], [], []
    for year in np.arange(years.start, years.stop):
        z = xr.open_dataset(datapath + zfilename.format(year=year), chunks={'time': 100})
        u = xr.open_dataset(datapath + ufilename.format(year=year), chunks={'time': 100})
        v = xr.open_dataset(datapath + vfilename.format(year=year), chunks={'time': 100})
        pr = xr.open_dataset(datapath + prfilename.format(year=year), chunks={'time': 100})
        z = z.assign_coords(longitude=(z.coords['longitude'].values + 180) % 360 - 180)
        u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
        v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
        pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
        u = u.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        z = z.sel(latitude=slice(20, -70), longitude=slice(-160, -10))
        v = v.sel(latitude=slice(20, -60), longitude=slice(-80, -10))
        pr = pr.sel(latitude=slice(20, -60), longitude=slice(-80, -10))

        z = z.to_array().isel(variable=0).drop('variable')
        u = u.to_array().isel(variable=0).drop('variable')
        v = v.to_array().isel(variable=0).drop('variable')
        pr = pr.to_array().isel(variable=0).drop('variable')
        u_list.append(u)
        v_list.append(v)
        z_list.append(z)
        pr_list.append(pr)
        print(str(year))
    u = xr.concat(u_list, dim='time')
    v = xr.concat(v_list, dim='time')
    z = xr.concat(z_list, dim='time')
    pr = xr.concat(pr_list, dim='time')
    [x.close() for x in u_list]
    [x.close() for x in v_list]
    [x.close() for x in z_list]
    [x.close() for x in pr_list]
    u_list = None
    v_list = None
    z_list = None
    pr_list = None
    return u, v, z, pr

# ---- Loading paths and contants ---- #
print('\n #---- Loading paths ----# \n')
outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
experiment = 'experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4/'
datapath = '/gws/nopw/j04/primavera1/observations/ERA5/'
vfilename = 'viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
ufilename = 'viwve_ERA5_6hr_{year}010100-{year}123118.nc'
zfilename = 'zg_250_ERA5_6hrPlevPt_{year}010100-{year}123118.nc'
prfilename = 'pr_ERA5_6hr_{year}010100-{year}123118.nc'
basins = ['Tiete', 'Uruguai']
seasons = [1, 3]

# ---- Lazily read data ---- #
print('\n #---- Reading data lazily ----# \n')
years = slice(1981, 2010)
u, v, z, pr = read_u_v(years)
MAG = xr.open_dataset('~/phdscripts/phdlib/convlib/data/xarray_mair_grid_basins.nc')
MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})
with open(outpath + experiment + 'config.txt') as file:
    config = eval(file.read())
days = config['lcs_time_len'] / 4
files = [f for f in glob.glob(outpath + experiment + "**/partial_ridges_0*.nc", recursive=True)]
da = xr.open_mfdataset(files, preprocess=lambda x: x.sortby('time'))
da = da.to_array()
da = da.sortby('time')
da = da.isel(variable=0).drop('variable')

# ---- Laily calculate flux across Amazon borders ---- #
print('\n #---- Laily calculate flux across Amazon borders ----# \n')

dlat = np.sign(MAG['amazon'].diff('latitude'))
dlon = np.sign(MAG['amazon'].diff('longitude'))
border_amazon = np.abs(dlat) + np.abs(dlon)
border_amazon = border_amazon.where(border_amazon <= 1, 1)
influx = dlat * v + dlon * u
influx = influx.where(border_amazon)

# ---- Reading basin masks and calculating area ---- #
print('\n #----Reading basin masks and calculating area ----# \n')

assert pd.Index(da.time.values).duplicated().sum() == 0, 'There are repeated time indices'
mask_tiete = MAG['Tiete'].interp(latitude=da.latitude, longitude=da.longitude, method='nearest')
mask_uruguai = MAG['Uruguai'].interp(latitude=da.latitude, longitude=da.longitude, method='nearest')
mask_tiete = mask_tiete.sel(latitude=slice(-30, -15), longitude=slice(-55, -30))
mask_uruguai = mask_uruguai.sel(latitude=slice(-35, -15), longitude=slice(-65, -30))
area_uruguai = mask_uruguai.sum()
area_tiete = mask_tiete.sum()

# ---- time sel and season ---- #
pr = pr.sel(time=da.time)
u = u.sel(time=da.time)
v = v.sel(time=da.time)
z = z.sel(time=da.time)
influx = influx.sel(time=da.time)
#
# pr_season = pr.groupby('time.season').mean('time')
# u_season = u.groupby('time.season').mean('time')
# v_season = v.groupby('time.season').mean('time')
# z_season = z.groupby('time.season').mean('time')
# u_var_season = u.groupby('time.season').var('time')
# v_var_season = v.groupby('time.season').var('time')
# z_var_season = z.groupby('time.season').var('time')
# pr_var_season = pr.groupby('time.season').var('time')
#
# # ---- Loading in mem ---- #
#
# with ProgressBar():
#     pr_season = pr_season.load()
# with ProgressBar():
#     u_season = u_season.load()
# with ProgressBar():
#     v_season = v_season.load()
# with ProgressBar():
#     z_season = z_season.load()
# with ProgressBar():
#     u_var_season = u_var_season.load()
# with ProgressBar():
#     v_var_season = v_var_season.load()
# with ProgressBar():
#     pr_var_season = pr_var_season.load()
# with ProgressBar():
#     z_var_season = z_var_season.load()
#
# # ---- Saving netcdfs ---- #
#
# pr_season.name = 'pr_season'
# u_season.name = 'u_season'
# v_season.name = 'v_season'
# z_season.name = 'z_season'
# pr_var_season.name = 'pr_var_season'
# u_var_season.name = 'u_var_season'
# v_var_season.name = 'v_var_season'
# z_var_season.name = 'z_var_season'
# arrays = [pr_season, u_season, v_season, z_season, u_var_season, v_var_season, pr_var_season,
#           z_var_season]
# ds = xr.merge(arrays)
# ds.to_netcdf('~/ds_seasonal_avgs.nc')
ds = xr.open_dataset('~/ds_seasonal_avgs.nc')
masks = [mask_tiete, mask_uruguai]
seasons = ['DJF', 'JJA']
season_mask = [(pd.Timestamp(x).month % 12 + 3) // 3 for x in da.time.values]
season_mask = pd.Index(season_mask).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})
da['season'] = ('time'), season_mask

anomalies_basins = []
anomalies_basins_days_of_cz = []
for season in seasons:
    print('Season: ' + str(season))
    anomalies_basins_season = []
    anomalies_basins_seasons_days_of_cz = []

    for mask in masks:
        area = mask.sum()
        da_ = da * mask
        da_ts = da_.sum(['latitude', 'longitude'])
        with ProgressBar():
            da_ts = da_ts.load()
        da_ts = da_ts.where(da_ts.season == season, drop=True)
        days_of_cz = da_ts.where(da_ts/area > 0.05, drop=True)
        anomalies_basins_seasons_days_of_cz.append(days_of_cz)
        u_cz = u.sel(time=days_of_cz.time.values)
        v_cz = v.sel(time=days_of_cz.time.values)
        z_cz = z.sel(time=days_of_cz.time.values)
        z_cz_3 = z.shift(time=12).sel(time=days_of_cz.time.values)
        pr_cz = pr.sel(time=days_of_cz.time.values)
        u_cz = u.sel(time=days_of_cz.time.values)
        pr_cz = pr_cz.mean('time')
        u_cz = u_cz.mean('time')
        v_cz = v_cz.mean('time')
        z_cz = z_cz.mean('time')
        z_cz_3 = z_cz_3.mean('time')
        with ProgressBar():
            pr_cz = pr_cz.load()
        with ProgressBar():
            u_cz = u_cz.load()
        with ProgressBar():
            v_cz = v_cz.load()
        with ProgressBar():
            z_cz = z_cz.load()
        with ProgressBar():
            z_cz_3 = z_cz_3.load()
        anom_pr = pr_cz - ds.pr_season.sel(season=season)
        anom_u = u_cz - ds.u_season.sel(season=season)
        anom_v = v_cz - ds.v_season.sel(season=season)
        anom_z = z_cz - ds.z_season.sel(season=season)
        anom_z_3 = z_cz_3 - ds.z_season.sel(season=season)
        anom_pr.name = 'pr'
        anom_z.name = 'z'
        anom_z_3.name = 'z3'
        anom_u.name = 'u'
        anom_v.name = 'v'

        ds_anomalies = xr.merge([
            anom_pr,
            anom_u,
            anom_v,
            anom_z,
            anom_z_3
        ])
        anomalies_basins_season.append(ds_anomalies)
    anomalies_basins.append(anomalies_basins_season)
    anomalies_basins_days_of_cz.append(anomalies_basins_seasons_days_of_cz)


da_list = []
days_of_cz_list = []


for i in range(len(anomalies_basins)):
        da_list.append(
            xr.concat(anomalies_basins[i], dim=pd.Index(basins, name='basin'))
        )
        nlist = []
        for dd in anomalies_basins_days_of_cz[i]:
            nlist.append(dd.copy(data=np.ones(dd.shape)).sum('time'))

        days_of_cz_list.append(
            xr.concat(nlist, dim=pd.Index(basins, name='basin'))
        )

daa = xr.concat(da_list, dim='season')
daa_days_of_cz = xr.concat(days_of_cz_list, dim=pd.Index(seasons, name='season'))
daa.to_netcdf('~/basin_composites.nc')
daa_days_of_cz.to_netcdf('~/basin_composites_days_of_cz.nc')



    #  # Time series analysis - hidden for now
    #
    # days_of_cz = days_of_cz.copy(data=np.ones(days_of_cz.shape))
    #
    # season_mask = [(pd.Timestamp(x).month % 12 + 3) // 3 for x in pr.time.values]
    # pr_tiete = pr * mask_tiete
    # pr_tiete['season_mask'] = ('time'), season_mask
    # pr_tiete = pr_tiete.sum(['latitude', 'longitude'])
    # pr_tiete = pr_tiete.where(pr_tiete.season_mask == 1, drop=True)
    # pr_tiete = pr_tiete.load()
    #
    # plt.style.use('bmh')
    # fig, ax = plt.subplots(1, 1)
    # ax2 = ax.twinx()
    # (pr_tiete * 3600 * 6 / area_tiete).resample(time='Y').mean().plot(ax=ax, color='blue')
    # days_of_cz.resample(time='Y').sum().plot(ax=ax2, color='red')
    # ax2.set_ylabel('Number of CZ events ', color='red')
    # ax.set_ylabel('Average DJF rainfall', color='blue')
    #
    # ndays_total = pr_tiete.copy(data=np.ones(pr_tiete.shape))
    # pr_tiete_cz = pr_tiete.sel(time=days_of_cz.time)
    #
    # fig, ax = plt.subplots(1, 1)
    # (100 * pr_tiete_cz.resample(time='Y').sum() / pr_tiete.resample(time='Y').sum()).plot(ax=ax)
    # (100 * days_of_cz.resample(time='Y').sum()/ndays_total.resample(time='Y').sum()).plot(ax=ax)
    # plt.legend(['Fraction of rainfall in CZ days', 'Fraction of CZ days'])
    # plt.show()
    # ridges = xr.open_dataarray(outpath + experiment + 'ridges.nc')
    # da = da * ridges.sel(time=da.time)

