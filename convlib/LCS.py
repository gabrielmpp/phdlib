import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from meteomath import to_cartesian
from typing import Tuple
from meteomath import interpolate_c_stagger

# import matplotlib.pyplot as plt
from xarray import DataArray

LCS_TYPES = ['attracting', 'repelling']


class LCS:
    """
    Methods to compute LCS in 2D wind fields in xarrray dataarrays
    """

    def __init__(self, lcs_type: str, timestep: float = 1, dataarray_template=None, timedim='time'):
        assert isinstance(lcs_type, str), "Parameter lcs_type expected to be str"
        assert lcs_type in LCS_TYPES, f"lcs_type {lcs_type} not available"
        self.lcs_type = lcs_type
        self.timestep = timestep
        self.timedim = timedim
        self.dataarray_template = dataarray_template

    def __call__(self, ds: xr.Dataset = None, u: xr.DataArray = None, v: xr.DataArray = None) -> xr.DataArray:

        """
        :param ds: xarray dataset containing u and v as variables
        :param u: xarray datarray containing u-wind component
        :param v: xarray dataarray containing v-wind component
        :param timestep: float
        :return: xarray dataarray of eigenvalue
        """
        timestep = self.timestep

        if isinstance(ds, xr.Dataset):
            u = ds.u
            v = ds.v
        if isinstance(self.dataarray_template, xr.DataArray):
            template = self.dataarray_template
            u = xr.DataArray(u, coords=template.coords, dims=template.dims)
            v = xr.DataArray(v, coords=template.coords, dims=template.dims)

        u_dims = u.dims
        v_dims = v.dims

        assert set(u_dims) == set(v_dims), "u and v dims are different"
        assert set(u_dims) == {'latitude', 'longitude', timedim}, 'array dims should be latitude and longitude only'

        if not (hasattr(u, "x") and hasattr(u, "y")):
            print("Ascribing x and y coords do u")
            u = to_cartesian(u)
        if not (hasattr(v, "x") and hasattr(v, "y")):
            print("Ascribing x and y coords do v")
            v = to_cartesian(v)

        print("*---- Computing deformation tensor ----*")
        def_tensor = self._compute_deformation_tensor(u, v, timestep)
        def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
        def_tensor = def_tensor.dropna(dim='points')
        print("*---- Computing eigenvalues ----*")
        eigenvalues = xr.apply_ufunc(lambda x: self._compute_eigenvalues(x), def_tensor.groupby('points'))
        eigenvalues = eigenvalues.unstack('points')
        eigenvalues = eigenvalues.isel(derivatives=0).drop('derivatives')
        return eigenvalues

    def _compute_eigenvalues(self, def_tensor: np.array) -> np.array:
        d_matrix = def_tensor.reshape([2, 2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)

        if self.lcs_type == 'repelling':
            eigenvalues = max(np.linalg.eig(cauchy_green.reshape([2, 2]))[0])
        elif self.lcs_type == 'attracting':
            eigenvalues = min(np.linalg.eig(cauchy_green.reshape([2, 2]))[0])
        else:
            raise ValueError("lcs_type {lcs_type} not supported".format(lcs_type=self.lcs_type))

        eigenvalues = np.repeat(eigenvalues, 4).reshape(
            [4, 1])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
        return eigenvalues

    @staticmethod
    def _parcel_propagation(u, v, timestep, propdim="time"):
        """

        :param u:
        :param v:
        :return:
        """

        times = u[propdim].values.tolist()
        if timestep < 0:
            times.reverse()  # inplace

        # initializing and integrating
        y_futur = u.y
        x_futur = u.x
        for time in times:

            print(f'Propagating time {time}')

            y_buffer = y_futur + timestep * v.sel({propdim: time}).interp(y=y_futur, x=x_futur, kwargs={'fill_value': None})
            x_buffer = x_futur + timestep * u.sel({propdim: time}).interp(y=y_futur, x=x_futur, kwargs={'fill_value': None})
            x_futur = x_buffer
            y_futur = y_buffer
        return x_futur, y_futur

    def _compute_deformation_tensor(self, u: xr.DataArray, v: xr.DataArray, timestep: float) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the meridional wind field
        :param timestep: float
        :return: xr.DataArray of the deformation tensor
        """

        x_futur, y_futur = self._parcel_propagation(u, v, timestep)
        u, v, eigengrid = interpolate_c_stagger(u, v)
        dx = u.x.diff('longitude')
        dy = u.y.diff('latitude')
        dxdx = x_futur.diff('longitude')/dx
        dxdy = x_futur.diff('latitude')/dy
        dydy = y_futur.diff('latitude')/dy
        dydx = y_futur.diff('longitude')/dx

        dxdx = dxdx.transpose('latitude', 'longitude').drop('x').drop('y')
        dxdy = dxdy.transpose('latitude', 'longitude').drop('x')
        dydy = dydy.transpose('latitude', 'longitude').drop('x').drop('y')
        dydx = dydx.transpose('latitude', 'longitude').drop('y')
        dxdx.name = 'dxdx'
        dxdy.name = 'dxdy'
        dydy.name = 'dydy'
        dydx.name = 'dydx'

        def_tensor = xr.merge([dxdx, dxdy, dydx, dydy])
        def_tensor = def_tensor.to_array()
        def_tensor = def_tensor.rename({'variable': 'derivatives'})
        def_tensor = def_tensor.transpose('derivatives', 'latitude', 'longitude')
        print(def_tensor)
        raise ValueError("STOP")
        return def_tensor

    def _compute_deformation_tensor_old(self, u: xr.DataArray, v: xr.DataArray, timestep: float) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the meridional wind field
        :param timestep: float
        :return: xr.DataArray of the deformation tensor
        """

        u, v, eigengrid = interpolate_c_stagger(u, v)

        # ------- Computing dy_futur/dx -------- #
        # When derivating with respect to x we use u coords in the Arakawa C grid
        y_futur = u.y + v.interp(latitude=u.latitude, longitude=u.longitude,
                                 kwargs={'fill_value': None}) * timestep  # fill_value=None extrapolates
        dx = u.x.diff('longitude')
        dydx = y_futur.diff('longitude') / dx
        dydx['longitude'] = eigengrid.longitude

        # ------- Computing dx_futur/dx -------- #
        x_futur = u.x + u * timestep
        dx = u.x.diff('longitude')
        dxdx = x_futur.diff('longitude') / dx
        dxdx['longitude'] = eigengrid.longitude

        # ------- Computing dx_futur/dy -------- #
        # When derivating with respect to y we use v coords in the Arakawa C grid
        x_futur = v.x + u.interp(latitude=v.latitude, longitude=v.longitude,
                                 kwargs={'fill_value': None}) * timestep  # fill_value=None extrapolates
        dy = v.y.diff('latitude')
        dxdy = x_futur.diff('latitude') / dy
        dxdy['latitude'] = eigengrid.latitude

        # ------- Computing dy_futur/dy -------- #
        y_futur = v.y + v * timestep
        dy = v.y.diff('latitude')
        dydy = y_futur.diff('latitude') / dy
        dydy['latitude'] = eigengrid.latitude

        dxdx = dxdx.transpose('latitude', 'longitude').drop('x').drop('y')
        dxdy = dxdy.transpose('latitude', 'longitude').drop('x')
        dydy = dydy.transpose('latitude', 'longitude').drop('x').drop('y')
        dydx = dydx.transpose('latitude', 'longitude').drop('y')
        dxdx.name = 'dxdx'
        dxdy.name = 'dxdy'
        dydy.name = 'dydy'
        dydx.name = 'dydx'

        def_tensor = xr.merge([dxdx, dxdy, dydx, dydy])
        def_tensor = def_tensor.to_array()
        def_tensor = def_tensor.rename({'variable': 'derivatives'})
        def_tensor = def_tensor.transpose('derivatives', 'latitude', 'longitude')
        return def_tensor

def find_maxima(eigenarray: xr.DataArray):
    data = eigenarray.values
    neighborhood_size = 4
    threshold = 0.08

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(diff)
    diff_array = xr.DataArray(diff, coords=eigenarray.coords, dims=eigenarray.dims)
    out_array = xr.DataArray(data_max, coords=eigenarray.coords, dims=eigenarray.dims)
    labeled_array = xr.DataArray(labeled, coords=eigenarray.coords, dims=eigenarray.dims)
    return out_array, diff_array, labeled_array


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs

    # u_path = '/path/to/ncdf/files/u.nc'
    # v_path = '/path/to/ncdf/files/v.nc'
    # u = xr.open_dataarray(u_path)
    # v = xr.open_dataarray(v_path)
    # lcs = LCS()
    # eigenvalues: xr.DataArray  # type hint
    # eigenvalues = lcs(u, v)
    eigenarray = xr.open_dataarray('data/SL.nc')

    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')

    for time in eigenarray.time.values:
        f, ax = plt.subplots(1, 1, figsize=(14, 14), subplot_kw={'projection': ccrs.PlateCarree()})
        eigenarray.sel(time=time).plot.contourf(cmap='nipy_spectral', vmax=0.7, ax=ax, levels=50)
        ax.coastlines(color='white')
        plt.savefig(f'tempfigs/eigenvalues_{time}.png')
        plt.close()

    for time in eigenarray.time.values:
        f, axarr = plt.subplots(1, 3, figsize=(3 * 14, 14), subplot_kw={'projection': ccrs.PlateCarree()})
        max_array, diff_array, labeled_array = find_maxima(eigenarray.sel(time=time))

        max_array.plot(ax=axarr[0], vmax=1, cmap='nipy_spectral')
        diff_array.plot(ax=axarr[1])
        labeled_array.where(labeled_array != 0).plot(ax=axarr[2], cmap='tab20')
        axarr[0].coastlines(color='white')
        axarr[1].coastlines()
        axarr[2].coastlines()

        plt.savefig(f'tempfigs/labeled_{time}.png')
        plt.close()
    print(None)
