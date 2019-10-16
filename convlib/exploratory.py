import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import meteomath
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':

    print("*---- Reading arrays ----*")
    u_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwve_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 100))
    v_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwvn_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 100))
    pr_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/pr_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 100))

    tcwv_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/tcwv_ERA5_6hr_2000010100-2000123118.nc').isel(time=slice(None, 100))
    array_full = xr.open_dataarray('/group_workspaces/jasmin4/upscale/gmpp/convzones/SL_repelling_2000_lcstimelen_1.nc').isel(time=slice(None, 100))
    array_full = xr.apply_ufunc(lambda x: np.log(x), array_full ** 0.5)
    print("*---- Transforming arrays ----*")
    pr_full.coords['longitude'].values = (pr_full.coords['longitude'].values + 180) % 360 - 180

    u_full.coords['longitude'].values = (u_full.coords['longitude'].values + 180) % 360 - 180
    v_full.coords['longitude'].values = (v_full.coords['longitude'].values + 180) % 360 - 180
    tcwv_full.coords['longitude'].values = (tcwv_full.coords['longitude'].values + 180) % 360 - 180
    u_full = u_full.sel(latitude=array_full.latitude, longitude=array_full.longitude)
    v_full = v_full.sel(latitude=array_full.latitude, longitude=array_full.longitude)
    pr_full = pr_full.sel(latitude=array_full.latitude, longitude=array_full.longitude) * 6 * 3600

    u_full = u_full/tcwv_full
    v_full = v_full/tcwv_full

    u_full = meteomath.to_cartesian(u_full)
    v_full = meteomath.to_cartesian(v_full)
    print("*---- Calculating div ----*")
    conv_moist = -meteomath.divergence(u_full*tcwv_full, v_full*tcwv_full)*6*3600
    conv = -meteomath.divergence(u_full, v_full)*6*3600

    time = 0
    array_full = array_full*tcwv_full

    vmin = array_full.quantile(0.1)
    vmax = array_full.quantile(0.95)
    vmin_div = conv.quantile(0.1)
    vmax_div = conv.quantile(0.95)
    vmin_pr = pr_full.quantile(0.1)
    vmax_pr = pr_full.quantile(0.95)

    i = 0
    print("*---- Start plotting ----*")
    for time in u_full.time.values:
        u = u_full.sel(time=time)
        v = v_full.sel(time=time)
        array = array_full.sel(time=time, method='pad')
        print(array)
        conv_array = conv.sel(time=time, method='pad')
        pr = pr_full.sel(time=time, method='pad')

        conv_moist_array = conv_moist.sel(time=time, method='pad')

        print(f'Plotting time {time}')
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[40, 40],
                                 subplot_kw={'projection': ccrs.Orthographic(-40, -20)})
        array.plot.contourf(levels=12, vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False,
                                        vmax=vmax, cmap='inferno', ax=axes[0, 0])
        axes[0, 0].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(),density=1)
        axes[0, 0].coastlines(color='red')
        axes[0, 0].set_title("FTLE * tcwv")

        conv_array.plot.contourf(levels=12,vmin=vmin_div, transform=ccrs.PlateCarree(), add_colorbar=False,
                                        vmax=vmax_div, cmap='inferno', ax=axes[0, 1])
        axes[0, 1].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axes[0, 1].coastlines(color='red')
        axes[0, 1].set_title("Conv. of the vert. scaled wind")

        pr.plot.contourf(levels=12,vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False,
                                        vmax=vmax, cmap='inferno', ax=axes[1, 0])
        axes[1, 0].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axes[1, 0].coastlines(color='red')
        axes[1, 0].set_title("6 hourly precipitation")

        im = conv_moist_array.plot.contourf(levels=12,vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False,
                                        vmax=vmax, cmap='inferno', ax=axes[1, 1])
        axes[1, 1].streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                           transform=ccrs.PlateCarree(), density=1)
        axes[1, 1].coastlines(color='red')
        axes[1, 1].set_title("Conv. of the vert. integ. wv flux")

        cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7], projection=ccrs.PlateCarree())
        fig.colorbar(im, cax=cbar_ax)
        fig.tight_layout()
        plt.savefig('tempfigs/comparison/{:02d}.png'.format(i))
        plt.close()
        i += 1
