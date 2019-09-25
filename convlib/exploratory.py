import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


if __name__ == '__main__':
    u = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwve_ERA5_6hr_2000010100-2000123118.nc')
    v = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwvn_ERA5_6hr_2000010100-2000123118.nc')
    mag = (u**2 + v**2)**0.5

    array = xr.open_dataarray('/group_workspaces/jasmin4/upscale/gmpp_convzones/SL_repelling_2000.nc')
    time = 0

    vmin = array.quantile(0.1)
    vmax = array.quantile(0.95)
    for time in array.time:
        print(f'Plotting time {time}')
        fig, ax = plt.subplots(figsize=[20,20],subplot_kw={'projection':ccrs.PlateCarree()})
        array.sel(time=time).plot(vmin=vmin, transform=ccrs.PlateCarree(),
                                        vmax=vmax, cmap='gray', ax=ax)
        ax.streamplot(x=u.longitude.values, y=u.latitude.values, u=u.sel(time=time).values, v=v.sel(time=time).values,
                       colors=mag.sel(time=time).values, density=0.2)
        ax.coastlines(color='red')
        plt.savefig(f'tempfigs/{time}.png')
