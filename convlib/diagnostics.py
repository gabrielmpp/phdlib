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


@jit(parallel=True)
def apply_binary_mask(times, dep_lat, dep_lon, mask, reverse=True):
    array_list = []
    origins = xr.zeros_like(mask)

    for i in numba.prange(times.shape[0]):
        time = times[i]
        dep_lat_, dep_lon_ = dep_lat.sel(time=time).copy(), dep_lon.sel(time=time).copy()
        dep_lat_nan = np.isnan(dep_lat_.values.flatten())
        dep_lon_nan = np.isnan(dep_lon_.values.flatten())
        assert (dep_lat_nan == dep_lon_nan).all(), "This should not happen!"

        dep_lat_no_nan = dep_lat_.values.flatten()[~dep_lat_nan]
        dep_lon_no_nan = dep_lon_.values.flatten()[~dep_lon_nan]

        points = [x for x in zip(dep_lat_no_nan, dep_lon_no_nan)]
        landsea = list()
        for point in points:
            landsea.append(
                mask.sel(latitude=point[0], longitude=point[1], method='nearest').values
            )
            origins.sel(latitude=point[0], longitude=point[1], method='nearest').values += 1
        vals = dep_lat_.values
        if reverse:
            vals[~np.isnan(vals)] = [0 if x == 1 else 1 for x in landsea]  # switching sea breeze to 1
        else:
            vals[~np.isnan(vals)] = [x for x in landsea]
        array_list.append(vals)
        print("Done time {}".format(time))
    return array_list, origins


def add_basin_coord(array, MAG):
    basin_names = list(MAG.coords.keys())
    [basin_names.remove(element) for element in ["lat", "lon", "points", "time"]]
    basin_avg = {}
    for basin in basin_names:
        array.coords[basin] = (("latitude", "longitude"), MAG.coords[basin].values)

    return array


def find_seabreezes(region, start_year, final_year, lcstimelen):
    years = range(start_year, final_year)

    # ---- File reading and formatting ---- #
    departures = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_departuretimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years, reverseLat=True).isel(time=slice(None, None))
    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
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

    dep_lat, dep_lon = dep_y.where(ftle_array > threshold, np.nan), dep_x.where(ftle_array > threshold, np.nan)
    # dep_lat = dep_y.copy(data=dep_lat)
    # dep_lon = dep_x.copy(data=dep_lon)
    dep_mask = xr.full_like(dep_x, 1)

    # Selecting only continental points (sort of)
    dep_lat = dep_lat.where(landSea_mask == 1, drop=True)
    dep_lon = dep_lon.where(landSea_mask == 1, drop=True)
    times = dep_lat.time.values
    array_list = apply_binary_mask(times, dep_lat, dep_lon, landSea_mask)

    seabreeze = dep_lat.copy(data=np.stack(array_list))
    breezetime = seabreeze.where(seabreeze == 1, drop=True).groupby('time.hour').sum('time').stack(
        points=['latitude', 'longitude']).groupby('points').apply(np.argmax).unstack()
    # seabreeze = dep_lat.groupby('time').apply(dummy)
    seabreeze.to_netcdf(
        f'/group_workspaces/jasmin4/upscale/gmpp/convzones/seabreeze_{start_year}_{final_year}_{region}_lcstimelen_{lcstimelen}.nc')
    return seabreeze


def seabreeze():
    region = 'NEBR'

    lcstimelen = 1
    vmin = 0
    vmax = 0.6
    start_year, final_year = 1995, 1996
    years = range(start_year, final_year)
    # seabreeze = xr.open_dataarray(f'/group_workspaces/jasmin4/upscale/gmpp/convzones/seabreeze_{start_year}_{final_year}_{region}.nc')

    seabreeze = find_seabreezes(region, start_year, final_year, lcstimelen)
    pr_array = read_nc_files(region=region,
                             basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                             filename='pr_ERA5_6hr_{year}010100-{year}123118.nc',
                             year_range=years, transformLon=True, reverseLat=True)

    pr_array = pr_array.sortby('latitude') * 6 * 3600  # total mm in 6h

    pr_array = pr_array.sel(latitude=seabreeze.latitude, longitude=seabreeze.longitude,
                            time=seabreeze.time, method='nearest')
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

    prec_frac = masked_precip.groupby('time.hour').sum('time') / pr_array.groupby('time.hour').sum('time')
    total = 4 * seabreeze.groupby('time.hour').sum('time') / seabreeze.time.values.shape[0]
    prec_frac = prec_frac / total
    # total = total - total.mean('hour')

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
    plt.savefig(f'tempfigs/seabreeze/seabreeze_{region}_lcstimelen_{lcstimelen}.png')
    plt.close()

    total.stack(points=['latitude', 'longitude']).mean('points').plot()
    plt.savefig('tempfigs/seabreeze/ciclo_diurno.png')


if __name__ == '__main__':

    region = 'SACZ'
    years = range(1985, 1990)
    lcstimelen = 6
    MAG = xr.open_dataset('~/phdlib/convlib/data/xarray_mair_grid_basins.nc')

    departures = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_departuretimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years)
    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years)
    u = read_nc_files(region=region,
                      basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                      filename='viwve_ERA5_6hr_{year}010100-{year}123118.nc',
                      year_range=years, transformLon=True, reverseLat=True,
                      time_slice_for_each_year=slice(lcstimelen-1, None))
    v = read_nc_files(region=region,
                      basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                      filename='viwvn_ERA5_6hr_{year}010100-{year}123118.nc',
                      year_range=years, transformLon=True, reverseLat=True,
                      time_slice_for_each_year=slice(lcstimelen-1, None))
    tcwv = read_nc_files(region=region,
                      basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                      filename='tcwv_ERA5_6hr_{year}010100-{year}123118.nc',
                      year_range=years, transformLon=True, reverseLat=True,
                      time_slice_for_each_year=slice(lcstimelen-1, None))
    raise ValueError(" a ")
    ftle = ftle_array.sel(latitude=MAG.lat.values, longitude=MAG.lon.values, method='nearest')
    u = u.sel(latitude=MAG.lat.values, longitude=MAG.lon.values, method='nearest')
    v = v.sel(latitude=MAG.lat.values, longitude=MAG.lon.values, method='nearest')
    tcwv = tcwv.sel(latitude=MAG.lat.values, longitude=MAG.lon.values, method='nearest')
    departures = departures.assign_coords(time=ftle.time.copy())

    departures = departures.sel(latitude=MAG.lat.values, longitude=MAG.lon.values, method='nearest')
    departures = add_basin_coord(MAG=MAG, array=departures)
    ftle = add_basin_coord(MAG=MAG, array=ftle)
    tcwv = add_basin_coord(MAG=MAG, array=tcwv)
    u = add_basin_coord(MAG=MAG, array=u)
    v = add_basin_coord(MAG=MAG, array=v)
    u = u/tcwv
    v = v/tcwv
    MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})

    basin='Tiete'
    basin_origin='amazon'
    CZ=0
    season='summer'
    avg_basin = ftle.where(ftle[basin] == 1).stack(points=['latitude', 'longitude']).sum('points')
    avg_basin = avg_basin.sel(time=departures.time)
    if season == 'summer':
        season_idxs = np.array([pd.to_datetime(x).month in [1, 2, 12] for x in departures.time.values])
    elif season == 'winter':
        season_idxs = np.array([pd.to_datetime(x).month in [5, 6, 7] for x in departures.time.values])


    avg_basin_season = avg_basin.sel(time=avg_basin.time[season_idxs])
    threshold = avg_basin_season.quantile(0.7)

    departures_basin = departures.where(departures[basin] == 1, drop=True)
    if CZ:
        dep_lat = departures_basin.y_departure.sel(time=departures_basin.time[season_idxs]).where(avg_basin_season > threshold)
        dep_lon = departures_basin.x_departure.sel(time=departures_basin.time[season_idxs]).where(avg_basin_season > threshold)
        u_basin = u.where(avg_basin_season > threshold).mean('time')
        v_basin = v.where(avg_basin_season > threshold).mean('time')
    else:
        dep_lat = departures_basin.y_departure.sel(time=departures_basin.time[season_idxs]).where(avg_basin_season < threshold)
        dep_lon = departures_basin.x_departure.sel(time=departures_basin.time[season_idxs]).where(avg_basin_season < threshold)
        u_basin = u.where(avg_basin_season < threshold).mean('time')
        v_basin = v.where(avg_basin_season < threshold).mean('time')

    masked, origins = apply_binary_mask(dep_lat.time.values.copy(), dep_lat=dep_lat.copy(), dep_lon=dep_lon.copy(),
                                        mask=MAG[basin_origin], reverse=False)

    departs = dep_lat.copy(data=np.stack(masked))
    ftle_basin = ftle.sel(time=dep_lat.time.values)

    magnitude = (u_basin**2 + v_basin**2)**0.5

    ### Plotting frequency of CZs that advected air parcels from the Amazon to Tiete
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[10, 10])
    gs = gridspec.GridSpec(1, 1, width_ratios=[1])
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    #ftle_basin.mean('time').sel(xrtools.createDomains('SACZ')).plot.contour(ax=ax, transform=ccrs.PlateCarree(),
    #    cmap='autumn', levels=10, vmax=ftle.quantile(0.9))
    #departs.mean('time').plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
    #    cmap='YlGnBu', levels=10, vmax=0.4, vmin=0)
    (origins.where(origins!=0).sel(xrtools.createDomains('SACZ'))/departs.time.values.shape[0]).plot(ax=ax, transform=ccrs.PlateCarree(),
        cmap='YlGnBu',  add_colorbar=True)
    ax.coastlines()
    MAG[basin].where(MAG[basin] == 1).where(MAG[basin] == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
                                                                                                 levels=[0.5],
                                                                                                 cmap='Gray',
                                                                                                 ax=ax)
    MAG['amazon'].where(MAG['amazon'] == 1).where(MAG['amazon'] == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
                                                                                                 levels=[0.5],
                                                                                                 cmap='Gray',
                                                                                                 ax=ax)
    #MAG.amazon.where(MAG.amazon == 1).where(MAG.amazon == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
    #                                                                                             levels=[0.5],
    #                                                                                             cmap='Gray',
    #
    #                                                                                             ax=ax)
    ax.streamplot(x=u_basin.longitude.values, y=v_basin.latitude.values, linewidth=0.3*magnitude.values,
                  u=u_basin.values, v=v_basin.values, color='darkgrey',
                  transform=ccrs.PlateCarree(), density=1)
    plt.savefig(f'tempfigs/basins/freqs_depts_{basin}_{season}_{region}_lcstimelen_{lcstimelen}_CZ_{CZ}.pdf')


    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[10, 10])
    gs = gridspec.GridSpec(1, 1, width_ratios=[1])
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    ftle_sacz_amazon_tiete = ftle.sel(time=departs.where(departs == 1, drop=True).time)
    u_sacz_amazon_tiete = u.sel(time=departs.where(departs == 1, drop=True).time).mean('time')
    v_sacz_amazon_tiete = v.sel(time=departs.where(departs == 1, drop=True).time).mean('time')
    magnitude = (u_sacz_amazon_tiete**2 + v_sacz_amazon_tiete**2)**0.5
    ftle_sacz_amazon_tiete.mean('time').plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
        cmap='YlGnBu', levels=10, vmax=ftle.quantile(0.9))
    strm = ax.streamplot(x=u_sacz_amazon_tiete.longitude.values, y=u_sacz_amazon_tiete.latitude.values, color=magnitude.values,
                  u=u_sacz_amazon_tiete.values, v=v_sacz_amazon_tiete.values,
                               transform=ccrs.PlateCarree(), density=1, cmap="autumn")
    MAG.Tiete.where(MAG.Tiete == 1).where(MAG.Tiete == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
                                                                                                 levels=[0.5],
                                                                                                 cmap='Gray',
                                                                                                 ax=ax)
    MAG.amazon.where(MAG.amazon == 1).where(MAG.amazon == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
                                                                                                 levels=[0.5],
                                                                                                 cmap='Gray',
                                                                                                 ax=ax)
    fig.colorbar(strm.lines)
    ax.coastlines()
    plt.savefig('tempfigs/basins/sacz_amazon_tiete.pdf')


    plt.close()
    ftle_summer = ftle.sel(time=ftle.time[season_idxs]).where(avg_basin_season > threshold)
    ftle_winter = ftle.sel(time=ftle.time[season_idxs]).where(avg_basin_season > threshold)

    fig = plt.figure(figsize=[40, 20])
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    axs = {}
    axs['summer'] = fig.add_subplot(gs[0, 0], projection=proj)
    axs['winter'] = fig.add_subplot(gs[0, 1], projection=proj)
    # Disable axis ticks
    for ax in axs.values():
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Add titles
    for name, ax in axs.items():
        ax.set_title(name)

    plot = ftle_summer.sel(createDomains(region)).where(avg_basin > avg_basin.quantile(0.8)).mean('time').plot.contourf(
        cmap='YlGnBu', levels=10, add_colorbar=False, ax=axs['summer'],
        vmax=ftle.quantile(0.9))

    MAG.Tiete.where(MAG.Tiete == 1).where(MAG.Tiete == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
                                                                                                 levels=[0.5],
                                                                                                 cmap='Gray',
                                                                                                 ax=axs['summer'])

    axs['summer'].coastlines(color='black')

    ftle_winter.sel(createDomains(region)).where(avg_basin > avg_basin.quantile(0.8)).mean('time').plot.contourf(
        cmap='YlGnBu', levels=10, add_colorbar=False, ax=axs['winter'],
        vmax=ftle.quantile(0.9))
    axs['winter'].coastlines(color='black')

    MAG.Tiete.where(MAG.Tiete == 1).where(MAG.Tiete == 1, 0).sel(createDomains(region)).plot.contour(add_colorbar=False,
                                                                                                 levels=[0.5],
                                                                                                 cmap='Gray',
                                                                                                 ax=axs['winter'])

    cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2], wspace=2.5)
    cax = fig.add_subplot(cbar_gs[0])
    plt.colorbar(plot, cax)

    plt.savefig('tempfigs/basins/summer.png')
    plt.close()
