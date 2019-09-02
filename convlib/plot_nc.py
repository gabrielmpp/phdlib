import xarray as xr
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
from skimage.morphology import skeletonize
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, canny

def detect_ridges(gray, sigma=0.5):
    hxx, hyy, hxy = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    return i2


def read_nc_files():
    years = range(1980, 1999)
    file_list = []
    for year in years:
        year = str(year)
        file_list.append(xr.open_dataarray(
            f'/group_workspaces/jasmin4/upscale/gmpp_convzones/SL_repelling_{year}.nc'))
    full_array = xr.concat(file_list, dim='time')
    return full_array


def plot_local():
    filepath_re = 'data/SL_repelling.nc'
    filepath_at = 'data/SL_attracting.nc'
    filepath_re_momentum = 'data/SL_repelling_momentum.nc'
    filepath_at_momentum = 'data/SL_attracting_momentum.nc'


    array_re = xr.open_dataarray(filepath_re)
    array_at = xr.open_dataarray(filepath_at)
    array_re_momentum = xr.open_dataarray(filepath_re_momentum)
    array_at_momentum = xr.open_dataarray(filepath_at_momentum)
    product = array_re**-1

    array_at1 = xr.apply_ufunc(lambda x: np.log(x), (array_at_momentum*array_re_momentum) ** 0.5)

    array_at2 = xr.apply_ufunc(lambda x: np.log(x), (array_re**-1) ** 0.5)
    array_at3 = xr.apply_ufunc(lambda x: np.log(x), (array_re_momentum**-1) ** 0.5)


    ridges = xr.apply_ufunc(lambda x: canny(x, sigma=2, low_threshold=0.4, use_quantiles=True), array_at2.groupby('time'))
    ridges_momentum = xr.apply_ufunc(lambda x: canny(x, sigma=2, low_threshold=0.4, use_quantiles=True), array_at3.groupby('time'))

    new_lon = np.linspace(array_at2.longitude[0].values, array_at2.longitude[-1].values, int(array_at2.longitude.values.shape[0] * 0.2))
    new_lat = np.linspace(array_at2.latitude[0].values, array_at2.latitude[-1].values, int(array_at2.longitude.values.shape[0] * 0.2))
    #array_at1 = array_at1.interp(latitude=new_lat, longitude=new_lon)
    #array_at2 = array_at2.interp(latitude=new_lat, longitude=new_lon)

    #array_at1 = array_at1.interp(latitude=array_at1.latitude, longitude=array_at1.longitude)

    # array_at1 = array_at
    # array_at2 = array_re**-1
    # array.isel(time=4).plot.contourf(cmap='RdBu', levels=100, vmin=0)
    for time in array_at1.time.values:
        f, axarr = plt.subplots(1, 3, figsize=(30, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        array_at1.sel(time=time).plot.contourf(levels=100, cmap='RdBu', transform=ccrs.PlateCarree(),
                                               ax=axarr[0])
        axarr[0].coastlines()
        array_at2.sel(time=time).plot.contourf(vmin=0,vmax=5,levels=100,cmap='nipy_spectral', transform=ccrs.PlateCarree(),
                                               ax=axarr[1])
        ridges.sel(time=time).plot.contour(cmap='Greys',ax=axarr[1])
        axarr[1].coastlines()
        array_at3.sel(time=time).plot.contourf(levels=100,cmap='nipy_spectral', transform=ccrs.PlateCarree(),
                                               ax=axarr[2])
        ridges_momentum.sel(time=time).plot.contour(cmap='Greys',ax=axarr[2])
        axarr[2].coastlines()
        # axarr.add_feature(states_provinces, edgecolor='gray')
        plt.savefig(f'./tempfigs/SL{time}.png')
        plt.close()


if __name__ == '__main__':
    # arr = read_nc_files()
    arr = xr.open_dataarray('/home/users/gmpp/out/SL_repelling_1980_1998.nc')
    array_mean = arr.groupby('time.month').mean('time')
    array_mean = xr.apply_ufunc(lambda x: np.log(x**0.5), array_mean)
    array_anomaly = xr.apply_ufunc(lambda x, y: x - y, array_mean, array_mean.mean('month') )
    max = array_anomaly.max()
    min = array_anomaly.min()
    for month in range(1,13):
        plt.figure(figsize=[10,10])
        array_anomaly.sel(month=month).plot(cmap='RdBu', vmax=0.8*max,
                                         vmin=0.8*min)
        plt.savefig(
            f'/home/users/gmpp/phdlib/convlib/tempfigs/sl_repelling_month_{month}.png'
        )
        plt.close()
