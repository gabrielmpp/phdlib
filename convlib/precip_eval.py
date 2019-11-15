import xarray as xr
import matplotlib as mpl
import sys
mpl.use('Agg')
from convlib.xr_tools import read_nc_files, createDomains
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import meteomath
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from convlib.xr_tools import xy_to_latlon
import cartopy.feature as cfeature
from xrtools import xrumap as xru
import pandas as pd
from numba import jit
import numba
import convlib.xr_tools as xrtools

if __name__ == '__main__':
    region = 'SACZ_big'
    years = range(2000, 2005)
    lcstimelen = 8
    basin = 'Tiete'
    season = 'DJF'

    basin_origin='amazon'
    MAG = xr.open_dataset('~/phdlib/convlib/data/xarray_mair_grid_basins.nc')
    mask = MAG[basin]

    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years, season=season, binary_mask=mask, lcstimelen=lcstimelen,
                               set_date=True)
    pr = read_nc_files(region=region,
                      basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                      filename='pr_ERA5_6hr_{year}010100-{year}123118.nc',
                      year_range=years, transformLon=True, reverseLat=True,
                      time_slice_for_each_year=slice(None, None), season=season, binary_mask=mask)
    ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)

    pr = pr * 1000
    pr_ts = pr.stack(points=['latitude', 'longitude']).mean('points')
    pr_ts = pr_ts.rolling(time=lcstimelen).mean().dropna('time')
    pr_ts = pr_ts.sel(time=ftle_array.time)
    ftle_ts = ftle_array.stack(points=['latitude', 'longitude']).mean('points')
    plt.figure(figsize=[10, 8])
    plt.style.use('fivethirtyeight')
    plt.scatter(x=pr_ts.values, y=ftle_ts.values)
    plt.xlabel('Precip (mm/6h)')
    plt.ylabel('FTLE')
    plt.savefig('tempfigs/diagnostics/precip.png')
    plt.close()

