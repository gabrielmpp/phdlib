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



if __name__ == '__main__':

    region = None
    years = range(2000, 2001)
    lcstimelen = 6
    MAG = xr.open_dataset('~/phdlib/convlib/data/xarray_mair_grid_basins.nc')
    lcstimelens = [1, 2, 4, 6]
    array_list = []
    for lcstimelen in lcstimelens:
        array_list.append(read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years))
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=[40, 20])
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])
    axs = dict()

    axs['6H'] = fig.add_subplot(gs[0, 0], projection=proj)
    axs['12H'] = fig.add_subplot(gs[0, 1], projection=proj)
    axs['24H'] = fig.add_subplot(gs[1, 0], projection=proj)
    axs['36H'] = fig.add_subplot(gs[1, 1], projection=proj)
    list_of_keys = ['6H', '12H', '24H', '36H']
    time = array_list[1].time.values[0]
    vmaxs = [2, 4, 8, 12]
    time_scaling = np.array([6., 12., 24., 36.])
    vmins = [0 for x in array_list]

    # Disable axis ticks
    for ax in axs.values():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Add titles
    for name, ax in axs.items():
        ax.set_title(name)

    for i in range(len(lcstimelens)):
        plot_array = xr.apply_ufunc(lambda x: time_scaling[i]**(-1) * np.log(x), array_list[i].sel(time=time) ** 0.5)

        ax = axs[list_of_keys[i]]

        plot = plot_array.plot.contourf(cmap='YlGnBu', levels=10, add_colorbar=False, ax=ax,
                                                          vmax=vmaxs[i], vmin=vmins[i])
        ax.coastlines()

    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(plot, cax)

    plt.savefig(f'tempfigs/daily_figs/{time}.png')
    plt.close()


