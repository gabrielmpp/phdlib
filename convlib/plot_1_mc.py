import xarray as xr
import matplotlib as mpl

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
import datetime


if __name__ == '__main__':

    region = None
    years = range(2000, 2001)
    lcstimelen = 6
    MAG = xr.open_dataset('~/phdlib/convlib/data/xarray_mair_grid_basins.nc')
    lcstimelens = [1, 4, 8, 16]
    array_list = []
    for lcstimelen in lcstimelens:
        array_list.append(read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years))
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=[25, 20])
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])
    axs = dict()

    axs['6H'] = fig.add_subplot(gs[0, 0], projection=proj)
    axs['1D'] = fig.add_subplot(gs[0, 1], projection=proj)
    axs['2D'] = fig.add_subplot(gs[1, 0], projection=proj)
    axs['4D'] = fig.add_subplot(gs[1, 1], projection=proj)
    list_of_keys = ['6H', '1D', '2D', '4D']
    time = '2000-02-10T12:00:00'
    vmaxs = [2, 2, 2, 2]
    time_scaling = (3600/86400)*6*np.array(lcstimelens)
    vmins = [0 for x in array_list]

    # Disable axis ticks
    for ax in axs.values():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)



    for i in range(len(lcstimelens)):
        plot_array = xr.apply_ufunc(lambda x: time_scaling[i]**(-1) * np.log(x),
                                    array_list[i].sel(time=pd.Timestamp(time)+pd.Timedelta(list_of_keys[i]))**0.5)

        ax = axs[list_of_keys[i]]

        plot = plot_array.sel(latitude=slice(-60,10), longitude=slice(-90,-30)).plot.contourf(cmap='Blues',vmin=0, vmax=2, levels=30, add_colorbar=False, ax=ax,
                                                          transform=ccrs.PlateCarree())
        plot_array.sel(latitude=slice(-60,10), longitude=slice(-90,-30)).plot.contour(levels=[1], color='black',ax=ax)
        ax.coastlines()

    # Add titles
    for name, ax in axs.items():
        ax.set_title(name)
    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(plot, cax)
    plt.tight_layout()

    plt.savefig(f'tempfigs/daily_figs/{time}.png')
    plt.close()


