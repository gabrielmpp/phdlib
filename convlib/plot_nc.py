import xarray as xr
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
from skimage.morphology import skeletonize
from typing import List
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, canny
from convlib.xr_tools import read_nc_files
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import pandas as pd

def detect_ridges(gray, sigma=0.5):
    hxx, hyy, hxy = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    return i2


def plot_local():
    filepath_re = 'data/SL_repelling.nc'
    filepath_at = 'data/SL_attracting.nc'
    filepath_re_momentum = 'data/SL_repelling_momentum.nc'
    filepath_at_momentum = 'data/SL_attracting_momentum.nc'

    array_re = xr.open_dataarray(filepath_re)
    array_at = xr.open_dataarray(filepath_at)
    array_re_momentum = xr.open_dataarray(filepath_re_momentum)
    array_at_momentum = xr.open_dataarray(filepath_at_momentum)
    product = array_re ** -1

    array_at1 = xr.apply_ufunc(lambda x: np.log(x), (array_at_momentum * array_re_momentum) ** 0.5)

    array_at2 = xr.apply_ufunc(lambda x: np.log(x), (array_re ** -1) ** 0.5)
    array_at3 = xr.apply_ufunc(lambda x: np.log(x), (array_re_momentum ** -1) ** 0.5)

    ridges = xr.apply_ufunc(lambda x: canny(x, sigma=2, low_threshold=0.4, use_quantiles=True),
                            array_at2.groupby('time'))
    ridges_momentum = xr.apply_ufunc(lambda x: canny(x, sigma=2, low_threshold=0.4, use_quantiles=True),
                                     array_at3.groupby('time'))

    new_lon = np.linspace(array_at2.longitude[0].values, array_at2.longitude[-1].values,
                          int(array_at2.longitude.values.shape[0] * 0.2))
    new_lat = np.linspace(array_at2.latitude[0].values, array_at2.latitude[-1].values,
                          int(array_at2.longitude.values.shape[0] * 0.2))
    # array_at1 = array_at1.interp(latitude=new_lat, longitude=new_lon)
    # array_at2 = array_at2.interp(latitude=new_lat, longitude=new_lon)

    # array_at1 = array_at1.interp(latitude=array_at1.latitude, longitude=array_at1.longitude)

    # array_at1 = array_at
    # array_at2 = array_re**-1
    # array.isel(time=4).plot.contourf(cmap='RdBu', levels=100, vmin=0)
    for time in array_at1.time.values:
        f, axarr = plt.subplots(1, 3, figsize=(30, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        array_at1.sel(time=time).plot.contourf(levels=100, cmap='RdBu', transform=ccrs.PlateCarree(),
                                               ax=axarr[0])
        axarr[0].coastlines()
        array_at2.sel(time=time).plot.contourf(vmin=0, vmax=5, levels=100, cmap='nipy_spectral',
                                               transform=ccrs.PlateCarree(),
                                               ax=axarr[1])
        ridges.sel(time=time).plot.contour(cmap='Greys', ax=axarr[1])
        axarr[1].coastlines()
        array_at3.sel(time=time).plot.contourf(levels=100, cmap='nipy_spectral', transform=ccrs.PlateCarree(),
                                               ax=axarr[2])
        ridges_momentum.sel(time=time).plot.contour(cmap='Greys', ax=axarr[2])
        axarr[2].coastlines()
        # axarr.add_feature(states_provinces, edgecolor='gray')
        plt.savefig(f'./tempfigs/SL{time}.png')
        plt.close()


domains = dict(
    AITCZ=dict(latitude=slice(-5, 15), longitude=slice(-50, -13)),
    SACZ=dict(latitude=slice(-40, -5), longitude=slice(-62, -20))
)

if __name__ == '__main__':
    region = "SACZ_small"
    lcs_time_len = 1

    arr = read_nc_files(region=region,
                        basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                        filename='SL_repelling_{year}' + f'_lcstimelen_{lcs_time_len}_v2.nc',
                        year_range=range(1990, 2000))
    # arr = xr.open_dataarray('/home/users/gmpp/out/SL_repelling_1980_1998.nc')
    sel_hour = 0
    hours = [pd.to_datetime(x).hour for x in arr.time.values]
    #arr = xr.apply_ufunc(lambda x: np.log(x), arr ** 0.5)

    array_mean = arr.where(np.array(hours).reshape(arr.shape[0], 1, 1) == sel_hour).groupby('time.season').mean('time')
    # array_mean = xr.apply_ufunc(lambda x: np.log(x**0.5), array_mean)
    # array_anomaly = xr.apply_ufunc(lambda x, y: x - y, array_mean, array_mean.mean('month'))
    # array_mean = array_anomaly # TODO just to plot var
    vmax = 0.8*array_mean.max()  # TODO REPLACE FOR ARRAY_ANOMALY
    vmin = array_mean.min()
    proj = ccrs.PlateCarree()


    seasons = array_mean.season.values
    fig = plt.figure(figsize=[20, 20])
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


    for season in seasons:
        plot = array_mean.sel(season=season).plot.contourf(ax=axs[season], transform=ccrs.PlateCarree(), add_colorbar=False,
                                         add_labels=False, levels=30, vmax=vmax, vmin=vmin, cmap='YlGnBu')
        axs[season].coastlines(color='black')
        axs[season].add_feature(cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces_lines', scale='50m',
            edgecolor='gray', facecolor='none'))
        axs[season].add_feature(cfeature.NaturalEarthFeature(
            'cultural', 'populated_places', scale='50m',
            edgecolor='black', facecolor='none'))
        axs[season].add_feature(cfeature.BORDERS)
    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(plot, cax)
    plt.savefig(f'tempfigs/seasonal/FTLE_{region}_lcstimelen_{lcs_time_len}_hour_{sel_hour}.png')
    plt.close()




    """
    for month in range(1, 13):
        print(f'Plotting month {month}')
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.Orthographic(-40, -20))
        array_mean.sel(month=month).plot.contourf(levels=10, cmap='RdBu',vmax=vmax,vmin=vmin,
                                                      ax=ax, transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        # TODO FIG IS NAMED VAR
        plt.savefig(
            '/home/users/gmpp/phdlib/convlib/tempfigs/sl_repelling_month_{month}_var_{region}_lcstimelen_{lcstimelen}.png'.format(
                month="{:02d}".format(month),
                region=region, lcstimelen=lcs_time_len)
        )
        plt.close()
    """