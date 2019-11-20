import xarray as xr
from convlib.xr_tools import read_nc_files, createDomains
import numpy as np

area = {
    'latitude': slice(-15, -10),
    'longitude': slice(-50, -38)
}
region="SACZ_big"
years = range(1995, 2005)
lcstimelen = 8  # 8
season = "DJF"
CZ = 1

ftle_array = read_nc_files(region=region,
                           basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                           filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                           year_range=years, season=season)

ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)

ftle_array = ftle_array.sel(area).groupby('time').mean()

ftle_array.to_netcdf('/home/users/gmpp/sacz_index.nc')
