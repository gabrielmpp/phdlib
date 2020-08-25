import glob

import xarray as xr

import pandas as pd


# season = sys.argv[1]
season=3
ERA5path = '/gws/nopw/j04/primavera1/observations/ERA5/'
FTLEpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4/'
files_rain = [f for f in glob.glob(ERA5path + "*pr_ERA5_6hr*.nc", recursive=True)]
files_ftle = [f for f in glob.glob(FTLEpath + "partial_0*.nc", recursive=True)]
da_rain = xr.open_mfdataset(files_rain)['tp']
da = xr.open_mfdataset(files_ftle, preprocess=lambda x: x.sortby('time'))
da = da.sortby('time')
da = da.to_array().isel(variable=0).drop('variable')
da_rain = da_rain.assign_coords(longitude=(da_rain.coords['longitude'].values + 180) % 360 - 180)
da_rain = da_rain.sel(time=slice(None, '2009'))
da_rain = da_rain.sel(longitude=slice(-140, -10))
da_rain = da_rain.sel(latitude=slice(15, -40))
# da = da.astype('float32')
# da_rain = da_rain.astype('float32')
da_rain = da_rain.sortby('longitude')
da_rain = da_rain.sortby('latitude')
da_rain = da_rain.sel(latitude=da.latitude, longitude=da.longitude, method='nearest')
da_rain = da_rain.assign_coords(latitude=da.latitude)
da_rain = da_rain.assign_coords(longitude=da.longitude)
da_rain = da_rain.sel(time=da.time)
season_mask = [(pd.Timestamp(x).month % 12 + 3) // 3 for x in da.time.values]
season_mask_rain = [(pd.Timestamp(x).month % 12 + 3) // 3 for x in da_rain.time.values]
da['season_mask'] = ('time'), season_mask
da_rain['season_mask'] = ('time'), season_mask_rain
da = da.where(da.season_mask==season, drop=True)
da_rain = da_rain.where(da_rain.season_mask==season, drop=True)
da = da.load()
da_rain = da_rain.load()
corr = xr.corr(da, da_rain, dim='time')
corr.to_netcdf(FTLEpath + f'corr_rain_{season}.nc')