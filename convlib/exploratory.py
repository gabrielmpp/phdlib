import xarray as xr
import matplotlib.pyplot as plt
from xrviz.dashboard import Dashboard
import cartopy.crs as ccrs


if __name__ == '__main__':
    u = xr.open_dataarray('/media/gabriel/gab_hd/data/sample_data/viwve_ERA5_6hr_2000010100-2000123118.nc')
    v = xr.open_dataarray('/media/gabriel/gab_hd/data/sample_data/viwvn_ERA5_6hr_2000010100-2000123118.nc')


    array = xr.open_dataarray('/media/gabriel/gab_hd/data/era5/convzones/SL_repelling_2000.nc')
    time=0
    vmin = array.quantile(0.2)
    vmax = array.quantile(0.99)
    fig, ax = plt.subplots(subplot_kw={'projection':ccrs.Mercator()})
    array.isel(time=time).plot.contourf(levels=100, vmin=vmin, vmax=vmax, cmap='gray', ax=ax)
    ax.coastlines()
    plt.show()
    dashboard = Dashboard(array)
    dashboard.show()