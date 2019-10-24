import xarray as xr
import matplotlib as mpl

mpl.use('Agg')
from convlib.xr_tools import read_nc_files
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import meteomath
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from convlib.xr_tools import xy_to_latlon

if __name__ == '__main__':
    region = 'SACZ'
    years = range(2000, 2008)
    if region == 'AITCZ':
        proj=ccrs.PlateCarree()
        figsize1 = (20,11)
        figsize2 = (30,20)
    elif region == 'SACZ':
        figsize1 = (20,11)
        figsize2 = (20,20)
        proj=ccrs.Orthographic(-40, -20)

    print('*---- Reading files ----*')
    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen_1.nc',
                               year_range=years)

    pr_array = read_nc_files(region=region,
                             basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                             filename='pr_ERA5_6hr_{year}010100-{year}123118.nc',
                             year_range=years, transformLon=True, reverseLat=True)
    pr_array = pr_array.sortby('latitude') * 6 * 3600  # total mm in 6h

    ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)

    threshold = ftle_array.quantile(0.8)
    pr_array = pr_array.sel(latitude=ftle_array.latitude, longitude=ftle_array.longitude, method='nearest')
    masked_precip = pr_array.where(ftle_array > threshold, 0)

    # ------ Total precip ------ #
    fig = plt.figure(figsize=figsize1)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.02])
    axs = {}

    axs['Total'] = fig.add_subplot(gs[0, 0], projection=proj)
    axs['Conv. zone events'] = fig.add_subplot(gs[0, 1], projection=proj)

    # Disable axis ticks
    for ax in axs.values():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Add titles
    for name, ax in axs.items():
        ax.set_title(name)
    vmin = 0
    vmax = pr_array.sum('time').quantile(0.95)

    masked_precip.sum('time').plot(ax=axs['Conv. zone events'], transform=ccrs.PlateCarree(), add_colorbar=False,
                                   add_labels=False, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
    total = pr_array.sum('time').plot(ax=axs['Total'], transform=ccrs.PlateCarree(), add_colorbar=False,
                                      add_labels=False,
                                      vmin=vmin, vmax=vmax, cmap='nipy_spectral')
    axs['Total'].coastlines(color='black')
    axs['Conv. zone events'].coastlines(color='black')
    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(total, cax)
    plt.savefig(f'tempfigs/diagnostics/total_{region}.png')
    plt.close()

    # ---- Seasonal plots ---- #
    fraction = masked_precip / pr_array
    fraction = fraction.groupby('time.season').mean('time')
    seasons = fraction.season.values

    fig = plt.figure(figsize=figsize2)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])
    axs = {}
    axs[seasons[0]] = fig.add_subplot(gs[0, 0], projection=proj)
    axs[seasons[1]] = fig.add_subplot(gs[0, 1], projection=proj)
    axs[seasons[2]] = fig.add_subplot(gs[1, 0], projection=proj)
    axs[seasons[3]] = fig.add_subplot(gs[1, 1], projection=proj)

    # Disable axis ticks
    for ax in axs.values():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Add titles
    for name, ax in axs.items():
        ax.set_title(name)
    vmin = 0
    vmax = fraction.quantile(0.99)

    for season in seasons:
        plot = fraction.sel(season=season).plot(ax=axs[season], transform=ccrs.PlateCarree(), add_colorbar=False,
                                                add_labels=False, vmin=vmin, vmax=vmax, cmap='nipy_spectral')
        axs[season].coastlines(color='black')

    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(plot, cax)
    plt.savefig(f'tempfigs/diagnostics/seasons_{region}.png')
    plt.close()
