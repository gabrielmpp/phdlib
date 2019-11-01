import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import meteomath
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from convlib.xr_tools import xy_to_latlon
if __name__ == '__main__':

    print("*---- Reading arrays ----*")
    u_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwve_ERA5_6hr_1990010100-1990123118.nc').isel(time=slice(None, 100))
    v_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwvn_ERA5_6hr_1990010100-1990123118.nc').isel(time=slice(None, 100))
    pr_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/pr_ERA5_6hr_1990010100-1990123118.nc').isel(time=slice(None, 100))

    tcwv_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/tcwv_ERA5_6hr_1990010100-1990123118.nc').isel(time=slice(None, 100))
    array_full = xr.open_dataarray('/group_workspaces/jasmin4/upscale/gmpp/convzones/SL_repelling_1990_lcstimelen_1_v2.nc').isel(time=slice(None, 100))
    #   array_full.time.values = tcwv_full.time.values #TODO do it properly
    #departures = xr.open_dataset('/group_workspaces/jasmin4/upscale/gmpp/convzones/SL_repelling_1990_departuretimelen_4_v2.nc').isel(time=slice(None, 100))
    array_full = xr.apply_ufunc(lambda x: np.log(x), array_full ** 0.5)
    print("*---- Transforming arrays ----*")
    pr_full.coords['longitude'].values = (pr_full.coords['longitude'].values + 180) % 360 - 180
    #departures.coords['longitude'].values = (departures.coords['longitude'].values + 180) % 360 - 180

    u_full.coords['longitude'].values = (u_full.coords['longitude'].values + 180) % 360 - 180
    v_full.coords['longitude'].values = (v_full.coords['longitude'].values + 180) % 360 - 180
    tcwv_full.coords['longitude'].values = (tcwv_full.coords['longitude'].values + 180) % 360 - 180
    u_full = u_full.sel(latitude=array_full.latitude, longitude=array_full.longitude)
    #departures = departures.sel(latitude=array_full.latitude, longitude=array_full.longitude)

    v_full = v_full.sel(latitude=array_full.latitude, longitude=array_full.longitude)
    pr_full = pr_full.sel(latitude=array_full.latitude, longitude=array_full.longitude) * 6 * 3600
    #dep_x, dep_y = departures.x_departure, departures.y_departure

    u_full = u_full/tcwv_full
    v_full = v_full/tcwv_full

    u_full = meteomath.to_cartesian(u_full)
    v_full = meteomath.to_cartesian(v_full)
    print("*---- Calculating div ----*")
    conv_moist = -meteomath.divergence(u_full*tcwv_full, v_full*tcwv_full)*6*3600
    conv = -meteomath.divergence(u_full, v_full) * 6 * 3600 * 10 #10 is just an avg tcwv

    time = 0
    array_moist = array_full*tcwv_full
    array_full = array_full*20

    vmin = array_moist.quantile(0.1)
    vmax = array_moist.quantile(0.95)
    vmin_div = conv.quantile(0.1)
    vmax_div = conv.quantile(0.95)
    vmin_pr = pr_full.quantile(0.1)
    vmax_pr = pr_full.quantile(0.95)
    #dep = xr.full_like(dep_x, 1)
    #dep_x.coords['time'] = array_moist.coords['time']
    #dep_y.coords['time'] = array_moist.coords['time']

    #dep_lat, dep_lon = xy_to_latlon(dep_x.where(array_moist > 3, np.nan).values, dep_y.where(array_moist > 10, np.nan).values)
    #dep_lat = dep_y.copy(data=dep_lat)
    #dep_lon = dep_x.copy(data=dep_lon)


    i = 0
    print("*---- Start plotting ----*")
    for time in u_full.time.values:
        u = u_full.sel(time=time)
        v = v_full.sel(time=time)
        array = array_full.sel(time=time, method='pad')
        array_m = array_moist.sel(time=time, method='pad')

        conv_array = conv.sel(time=time, method='pad')
        pr = pr_full.sel(time=time, method='pad')
        #dep_lat_array = dep_lat.sel(time=time, method='pad')
        #dep_lon_array = dep_lon.sel(time=time, method='pad')

        conv_moist_array = conv_moist.sel(time=time, method='pad')

        print(f'Plotting time {time}')
        fig = plt.figure(figsize=(40, 11))

        # Setup axes
        gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.08])
        axs = {}
        axs['FTLE Moist'] = fig.add_subplot(gs[:, :2], projection=ccrs.Orthographic(-40, -20))
        axs['FTLE Dry'] = fig.add_subplot(gs[0, 2], projection=ccrs.Orthographic(-40, -20))
        axs['Precip'] = fig.add_subplot(gs[1, 2], projection=ccrs.Orthographic(-40, -20))
        axs['Conv of vert. scaled wind'] = fig.add_subplot(gs[0, 3], projection=ccrs.Orthographic(-40, -20))
        axs['Conv of vert. int. wv flux'] = fig.add_subplot(gs[1, 3], projection=ccrs.Orthographic(-40, -20))
        # Disable axis ticks
        for ax in axs.values():
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        # Add titles
        for name, ax in axs.items():
            ax.set_title(name)

        plot_ftle = array_m.plot.contourf(levels=12, vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False,
                                        vmax=vmax, cmap='inferno', ax=axs['FTLE Moist'])
        axs['FTLE Moist'].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        #axs['FTLE Moist'].scatter(x=dep_lon_array.values, y=dep_lat_array.values, transform=ccrs.PlateCarree(), alpha=0.5,s=100)

        axs['FTLE Moist'].coastlines(color='red')

        array.plot.contourf(levels=12, vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False,
                                        vmax=vmax, cmap='inferno', ax=axs['FTLE Dry'])
        axs['FTLE Dry'].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axs['FTLE Dry'].coastlines(color='red')

        conv_array.plot.contourf(levels=12, vmin=vmin_div, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False,
                                        vmax=vmax_div, cmap='inferno', ax=axs['Conv of vert. scaled wind'])
        axs['Conv of vert. scaled wind'].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axs['Conv of vert. scaled wind'].coastlines(color='red')

        pr.plot.contourf(levels=12,vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False,add_labels=False,
                                        vmax=vmax, cmap='inferno', ax=axs['Precip'])
        axs['Precip'].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axs['Precip'].coastlines(color='red')

        conv_moist_array.plot.contourf(levels=12, vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False,add_labels=False,
                                        vmax=vmax, cmap='inferno', ax=axs['Conv of vert. int. wv flux'])
        axs['Conv of vert. int. wv flux'].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axs['Conv of vert. int. wv flux'].coastlines(color='red')
        # Colour bars
        cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 4], wspace=2.5)
        cax = fig.add_subplot(cbar_gs[0])
        plt.colorbar(plot_ftle, cax)

        #cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7], projection=ccrs.PlateCarree())
        #fig.colorbar(im, cax=cbar_ax)
        #fig.tight_layout()
        plt.savefig('tempfigs/comparison/{:02d}.png'.format(i))
        plt.close()
        i += 1
