import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


if __name__ == '__main__':
    u_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwve_ERA5_6hr_2000010100-2000123118.nc')
    v_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/viwvn_ERA5_6hr_2000010100-2000123118.nc')
    tcwv_full = xr.open_dataarray('/gws/nopw/j04/primavera1/observations/ERA5/tcwv_ERA5_6hr_2000010100-2000123118.nc')

    u_full.coords['longitude'].values = (u_full.coords['longitude'].values + 180) % 360 - 180
    v_full.coords['longitude'].values = (v_full.coords['longitude'].values + 180) % 360 - 180
    tcwv_full.coords['longitude'].values = (tcwv_full.coords['longitude'].values + 180) % 360 - 180
    u_full = u_full/tcwv_full
    v_full = v_full/tcwv_full

    array_full = xr.open_dataarray('/group_workspaces/jasmin4/upscale/gmpp/convzones/SL_repelling_2000_lcstimelen_1.nc')
    print(array_full)
    print(u_full)
    u_full = u_full.sel(latitude=array_full.latitude, longitude=array_full.longitude)
    v_full = v_full.sel(latitude=array_full.latitude, longitude=array_full.longitude)

    time = 0

    vmin = array_full.quantile(0.1)


    vmax = array_full.quantile(0.95)
    i=0
    for time in u_full.time.values:
        u = u_full.sel(time=time)
        v = v_full.sel(time=time)
        array = array_full.sel(time=time, method='pad')
        mag = (u ** 2 + v ** 2) ** 0.5
        print(f'Plotting time {time}')
        fig, ax = plt.subplots(figsize=[30,20],subplot_kw={'projection':ccrs.Orthographic(-40,-20)})
        array.plot.contourf(levels=12,vmin=vmin, transform=ccrs.PlateCarree(), add_colorbar=False,
                                        vmax=vmax, cmap='inferno', ax=ax)
        ax.streamplot(x=u.longitude.values, y=u.latitude.values, u=u.values, v=v.values, transform=ccrs.PlateCarree(),density=1)
                      # color=mag.values)
        ax.coastlines(color='red')
        plt.savefig(f'tempfigs/{i}.png')
        plt.close()
        i+=1
