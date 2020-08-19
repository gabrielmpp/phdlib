from scipy.signal import peak_widths, find_peaks
from skimage.filters import frangi, hessian
import glob
import xarray as xr
import numpy as np
import matplotlib
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
    'experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4'
]

experiment = experiments[0]
with open(outpath + experiment + '/config.txt') as file:
    config = eval(file.read())
days = config['lcs_time_len'] / 4
files_ridges = [f for f in glob.glob(outpath + experiment + "**/*ridges*.nc", recursive=True)]
files = [f for f in glob.glob(outpath + experiment + "**/partial_0*.nc", recursive=True)]
da = xr.open_mfdataset(files_ridges)
da = da.to_array().isel(variable=0).drop('variable')

ridges_seasons = da.groupby('time.season').sum('time')
ridges_seasons = ridges_seasons.load()
convert = 1/(4)  # Just to fix the season separation
ridges_seasons = ridges_seasons / (convert*da.time.shape[0])

cmap = cmr.freeze_r
ridges_seasons.name = 'Frequency of MAS events (%)'
p = (100 * ridges_seasons ).plot(col='season',robust=True, vmax=5,linewidth=0,antialiased=True,
                                     col_wrap=2, add_colorbar=True, cmap=cmap,
                                         subplot_kws={'projection': ccrs.PlateCarree()})

for ax in p.axes.flat:     ax.coastlines(color='black')
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
# for ax in p.axes.flat:
#     ax.set_edgecolor('face')


plt.savefig('tempfigs/new_frequency_season.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
p=(ridges_seasons.mean('season')*100).plot(robust=True, vmax=5,linewidth=0, antialiased=True,
                                         add_colorbar=True, cmap=cmap, ax=ax, transform=ccrs.PlateCarree())
ax.coastlines()
p.set_edgecolor('face')
gl = ax.gridlines(draw_labels=True)
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = ticker.FixedLocator([-80, -70, -60, -50, -40])
gl.xlines = False
gl.ylines = False
plt.savefig('tempfigs/new_frequency.png', dpi=600)
plt.close()

