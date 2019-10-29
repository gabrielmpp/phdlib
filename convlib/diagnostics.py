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
import cartopy.feature as cfeature
from xrtools import xrumap as xru
import pandas as pd


def precip_fraction(region, years):
    if region == 'AITCZ':
        proj = ccrs.PlateCarree()
        figsize1 = (20, 11)
        figsize2 = (30, 20)
    elif 'SACZ' in region:
        figsize1 = (20, 11)
        figsize2 = (20, 20)
        proj = ccrs.Orthographic(-40, -20)

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

    threshold = ftle_array.quantile(0.6)
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


def filter(region, years):
    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen_1.nc',
                               year_range=years)
    ftle_array = ftle_array.expand_dims('dummy')
    encoder = xru.autoencoder(n_components=7, dims_to_reduce=['dummy'], alongwith=['time'], mode='emd',
                              parallel=True).fit(ftle_array)
    transformed = encoder.transform(ftle_array)
    transformed = transformed.transpose('time', 'dim_to_groupby', 'encoded_dims')
    new_values = transformed.values.reshape(transformed.shape[0], ftle_array.shape[2], ftle_array.shape[3],
                                            transformed.shape[2])
    new_array = xr.DataArray(new_values, dims=['time', 'latitude', 'longitude', 'modes'],
                             coords=[transformed.time.values, ftle_array.latitude.values, ftle_array.longitude.values,
                                     transformed.encoded_dims.values])


def find_seabreezes(region, start_year, final_year):
    years = range(start_year, final_year)

    # ---- File reading and formatting ---- #
    departures = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_departuretimelen_1.nc',
                               year_range=years, reverseLat=True).isel(time=slice(None, None))
    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen_1.nc',
                               year_range=years).isel(time=slice(None, None))
    landSea_mask = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/landSea_mask.nc')
    landSea_mask = landSea_mask.squeeze('time').drop('time')
    landSea_mask.coords['longitude'] = (landSea_mask.coords['longitude'].values + 180) % 360 - 180
    landSea_mask = landSea_mask.sortby('latitude').sortby('longitude')
    landSea_mask = landSea_mask.interp_like(ftle_array, method='nearest')
    landSea_mask.values = landSea_mask.values.round(0)

    departures.coords['time'] = \
        pd.date_range(start=f"{start_year}-01-01T00:00:00",
                      periods=departures.time.shape[0], freq='6H')

    departures = departures.sortby('latitude')

    ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)
    threshold = ftle_array.quantile(0.6)

    dep_x, dep_y = departures.x_departure, departures.y_departure

    dep_lat, dep_lon = xy_to_latlon(dep_x.where(ftle_array > threshold, np.nan).values,
                                    dep_y.where(ftle_array > threshold, np.nan).values)
    dep_lat = dep_y.copy(data=dep_lat)
    dep_lon = dep_x.copy(data=dep_lon)
    dep_mask = xr.full_like(dep_x, 1)

    # Selecting only continental points (sort of)
    dep_lat = dep_lat.where(landSea_mask == 1, drop=True)
    dep_lon = dep_lon.where(landSea_mask == 1, drop=True)


    def dummy(x):
        time = x.time
        dep_lat_, dep_lon_ = dep_lat.sel(time=time), dep_lon.sel(time=time)
        dep_lat_nan = np.isnan(dep_lat_.values.flatten())
        dep_lon_nan = np.isnan(dep_lon_.values.flatten())
        assert (dep_lat_nan == dep_lon_nan).all(), "This should not happen!"

        dep_lat_no_nan = dep_lat_.values.flatten()[~dep_lat_nan]
        dep_lon_no_nan = dep_lon_.values.flatten()[~dep_lon_nan]

        points = [x for x in zip(dep_lat_no_nan, dep_lon_no_nan)]
        landsea = list()
        for point in points:
            landsea.append(
                landSea_mask.sel(latitude=point[0], longitude=point[1], method='nearest').values
            )
        vals = dep_lat_.values
        vals[~np.isnan(vals)] = [0 if x == 1 else 1 for x in landsea]  # switching sea breeze to 1
        dep_lat_.values = vals
        print("Done time {}".format(time))
        return dep_lat_


    seabreeze = dep_lat.groupby('time').apply(dummy)
    seabreeze.to_netcdf(f'/group_workspaces/jasmin4/upscale/gmpp/convzones/seabreeze_{start_year}_{final_year}_{region}.nc')

if __name__ == '__main__':
    region = 'NEBR'
    vmin = 0
    vmax = 0.6
    start_year, final_year = 2000, 2008
    years = range(start_year, final_year)
    seabreeze = xr.open_dataarray(f'/group_workspaces/jasmin4/upscale/gmpp/convzones/seabreeze_{start_year}_{final_year}_{region}.nc')

    #seabreeze = find_seabreezes(region, start_year, final_year)
    pr_array = read_nc_files(region=region,
                             basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                             filename='pr_ERA5_6hr_{year}010100-{year}123118.nc',
                             year_range=years, transformLon=True, reverseLat=True)
    pr_array = pr_array.sortby('latitude') * 6 * 3600  # total mm in 6h


    pr_array = pr_array.sel(latitude=seabreeze.latitude, longitude=seabreeze.longitude, method='nearest')
    masked_precip = pr_array.where(seabreeze == 1, 0)
    #
    # for time in seabreeze.time:
    #     fig = plt.figure(figsize=[10, 10])
    #     gs = gridspec.GridSpec(1, 1, width_ratios=[1])
    #
    #     axs = {'mask': fig.add_subplot(gs[0, 0], projection=proj)}
    #
    #     seabreeze.sel(time=time).plot(transform=ccrs.PlateCarree(), add_colorbar=True,
    #                                   ax=axs['mask'], vmin=0, vmax=1)
    #     axs['mask'].add_feature(cfeature.NaturalEarthFeature(
    #         'cultural', 'admin_1_states_provinces_lines', scale='50m',
    #         edgecolor='gray', facecolor='none'))
    #     axs['mask'].add_feature(cfeature.RIVERS)
    #     axs['mask'].add_feature(cfeature.COASTLINE)
    #     plt.savefig('tempfigs/seabreeze/{}.png'.format(time.values))
    #     plt.close()

    prec_frac = masked_precip.groupby('time.hour').sum('time')/pr_array.groupby('time.hour').sum('time')
    total = 4 * seabreeze.groupby('time.hour').sum('time') / seabreeze.time.values.shape[0]
    prec_frac = prec_frac/total
    #total = total - total.mean('hour')

    hours = total.hour.values
    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=[20, 20])
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])
    axs = {}
    axs[hours[0]] = fig.add_subplot(gs[0, 0], projection=proj)
    axs[hours[1]] = fig.add_subplot(gs[0, 1], projection=proj)
    axs[hours[2]] = fig.add_subplot(gs[1, 0], projection=proj)
    axs[hours[3]] = fig.add_subplot(gs[1, 1], projection=proj)

    # Disable axis ticks
    for ax in axs.values():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Add titles
    for name, ax in axs.items():
        ax.set_title(name)


    for hour in hours:
        plot = prec_frac.sel(hour=hour).plot.contourf(ax=axs[hour], transform=ccrs.PlateCarree(), add_colorbar=False,
                                         add_labels=False, levels=5, vmax=4, vmin=0, cmap='Blues')
        CS = total.sel(hour=hour).plot.contour(ax=axs[hour], levels=5, vmin=vmin, vmax=vmax, cmap="Reds")
        axs[hour].clabel(CS, inline=True, fontsize=15)
        axs[hour].coastlines(color='black')
        axs[hour].add_feature(cfeature.NaturalEarthFeature(
            'cultural', 'admin_1_states_provinces_lines', scale='50m',
            edgecolor='gray', facecolor='none'))
    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(plot, cax)
    plt.savefig(f'tempfigs/seabreeze/seabreeze_{region}.png')
    plt.close()

    total.stack(points=['latitude','longitude']).mean('points').plot()
    plt.savefig('tempfigs/seabreeze/ciclo_diurno.png')



    '''
    array_list = []

    def mask_f(x):
        time = x.time
        not_nans = ~np.isnan(dep_lat.sel(time=time).values)
        kwargs = dict(latitude=dep_lat.sel(time=time).values[not_nans].round(2),
                      longitude=dep_lon.sel(time=time).values[not_nans].round(2))
        kwargs = pd.DataFrame(kwargs)
        kwargs.drop_duplicates(inplace=True)
        kwargs = kwargs.to_dict(orient='list')
        kwargs['method'] = 'nearest'

        foo = x.sel(**kwargs, drop=True)
        x = x.loc[foo.latitude.values, foo.longitude.values]

        print("Done time {}".format(time.values))
        array_list.append(x)
        return x

    #dep_mask = dep_mask.stack(points=['latitude','longitude'])

    dep_mask = dep_mask.groupby('time').apply(mask_f)
    '''
