import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


if __name__ == '__main__':
    u_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwve_ERA5_6hr_2000010100-2000123118.nc')
    v_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwvn_ERA5_6hr_2000010100-2000123118.nc')

    array_full = xr.open_dataarray('/group_workspaces/jasmin4/upscale/gmpp_convzones/SL_repelling_2000.nc')
    time = 0

    vmin = array.quantile(0.1)
    vmax = array.quantile(0.95)
    for time in array.time:
        u = u_full.sel(time=time)
        v = v_full.sel(time=time)
        array = array_full.sel(time=time)
        mag = (u ** 2 + v ** 2) ** 0.5
        print(f'Plotting time {time}')
        fig, ax = plt.subplots(figsize=[20,20],subplot_kw={'projection':ccrs.PlateCarree()})
        array.plot(vmin=vmin, transform=ccrs.PlateCarree(),
                                        vmax=vmax, cmap='gray', ax=ax)
        ax.streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values,
                       colors=mag.values, density=0.2)
        ax.coastlines(color='red')
        plt.savefig(f'tempfigs/{time}.png')
