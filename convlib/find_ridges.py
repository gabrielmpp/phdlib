from scipy.signal import peak_widths, find_peaks
from skimage.filters import frangi, hessian
import glob
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmasher as cmr
outpath = '/group_workspaces/jasmin4/upscale/gmpp/convzones/'
experiments = [
    # 'experiment_timelen_12_d8513a9b-3fd6-4df4-9b05-377a9d8e64ca/',
    # 'experiment_timelen_16_105eee10-9804-4167-b151-9821c41136f6/',
    # 'experiment_timelen_4_5cc17190-9174-4642-b70c-a6170a808eb5/',
    'experiment_timelen_8_c102ec42-2a6f-4c98-be7d-31689c6c60a9/'
]

experiment = experiments[0]
with open(outpath + experiment + 'config.txt') as file:
    config = eval(file.read())
days = config['lcs_time_len'] / 4
files = [f for f in glob.glob(outpath + experiment + "**/*.nc", recursive=True)]
da = xr.open_dataarray(outpath + experiment + '/full_array.nc', chunks={'time': 100})
da = da.where(da > 0, 1e-6)
da = np.log(np.sqrt(da)) / days
da = da.resample(time='1D').mean()
# # It would take 5 hourse to run ridges below for the whole TS
# ridges = da.groupby('time').apply(hessian)
# ridges.to_netcdf(outpath + experiment + 'ridges.nc')
ridges = xr.open_dataarray(outpath + experiment + 'ridges.nc')
ridges = ridges.where(da > 0.8, 0)
ridges_seasons = ridges.groupby('time.season').mean('time')
ridges_seasons_anomaly = ridges_seasons - ridges_seasons.mean('season')
ridges_seasons = ridges_seasons.load()
ridges_seasons_anomaly = ridges_seasons_anomaly.load()
cmap = cmr.arctic_r
ridges_seasons.name = 'Frequency of MAS events (%)'
p = (ridges_seasons * 100).plot.contourf(vmin=0, vmax=40,col='season', levels=11,
                                     col_wrap=2, add_colorbar=True, cmap=cmap,
                                         subplot_kws={'projection': ccrs.PlateCarree()})

for ax in p.axes.flat:     ax.coastlines()
gl = p.axes.flat[0].gridlines(draw_labels=True)
gl.yformatter = LATITUDE_FORMATTER
gl.ylabels_right=False
gl.xlabels_top=False
gl.xlabels_bottom=False
gl.xlines = False
gl.ylines = False

gl = p.axes.flat[3].gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_left = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-80, -60, -40])
gl.xlines = False
gl.ylines = False


plt.savefig('tempfigs/new_frequency_season.pdf')
plt.close()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
(ridges_seasons.mean('season')*100).plot.contourf(levels=11, cmap=cmap, vmin=0, vmax=40, ax=ax)
ax.coastlines()

gl = ax.gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-80, -70, -60, -50, -40])
gl.xlines = False
gl.ylines = False
plt.savefig('tempfigs/new_frequency.pdf')
plt.close()

