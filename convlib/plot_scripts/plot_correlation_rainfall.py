import glob
from xr_tools.tools import common_index

import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import cartopy.crs as ccrs
import pandas as pd
import bottleneck
from xr_tools.tools import safely_read_multiple_files
from xr_tools.tools import latlonsel, spearman_correlation, spearman_pvalue
from scipy.signal import savgol_filter
import cmasher as cmr


ERA5path = '/gws/nopw/j04/primavera1/observations/ERA5/'
FTLEpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4/'
files_rain = [f for f in glob.glob(ERA5path + "*pr*.nc", recursive=True)]
files_ftle = [f for f in glob.glob(FTLEpath + "partial_0*.nc", recursive=True)]
da_rain = xr.open_mfdataset(files_rain)
da = xr.open_mfdataset(files_ftle)
da = da.to_array().isel(variable=0).drop('variable')


pr_ERA5_6hr_2020010100-2020123118.nc