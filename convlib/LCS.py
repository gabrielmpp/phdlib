
import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from meteomath import to_cartesian
from typing import Tuple
# import matplotlib.pyplot as plt

LCS_TYPES = ['attracting', 'repelling']


class LCS:
    """
    Methods to compute LCS in 2D wind fields in xarrray dataarrays
    """

    def __init__(self, lcs_type: str, timestep: float = 1, dataarray_template=None):
        assert isinstance(lcs_type, str), "Parameter lcs_type expected to be str"
        assert lcs_type in LCS_TYPES, f"lcs_type {lcs_type} not available"
        self.lcs_type = lcs_type
        self.timestep = timestep
        self.dataarray_template = dataarray_template

    def __call__(self, ds: xr.Dataset = None,  u: xr.DataArray = None, v: xr.DataArray = None) -> xr.DataArray:

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
        assert set(u_dims) == {'latitude', 'longitude'}, 'array dims should be latitude and longitude only'

        if not (hasattr(u, "x") and hasattr(u, "y")):
            print("Ascribing x and y coords do u")
            u = to_cartesian(u)
        if not (hasattr(v, "x") and hasattr(v, "y")):
            print("Ascribing x and y coords do v")
            v = to_cartesian(v)

        u, v, eigen_grid = interpolate_c_stagger(u, v)
        print("*---- Computing deformation tensor ----*")
        def_tensor = self._compute_deformation_tensor(u, v, eigen_grid, timestep)
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

        eigenvalues = np.repeat(eigenvalues, 4).reshape([4, 1])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
        return eigenvalues

    @staticmethod
    def _compute_deformation_tensor(u: xr.DataArray, v: xr.DataArray,
                                    eigengrid: xr.DataArray, timestep: float) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the meridional wind field
        :param timestep: float
        :return: xr.DataArray of the deformation tensor
        """

        # ------- Computing dy_futur/dx -------- #
        # When derivating with respect to x we use u coords in the Arakawa C grid
        y_futur = u.y + v.interp(latitude=u.latitude, longitude=u.longitude, kwargs={'fill_value': None}) * timestep #fill_value=None extrapolates
        dx = u.x.diff('longitude')
        dydx = y_futur.diff('longitude')/dx
        dydx['longitude'] = eigengrid.longitude

        # ------- Computing dx_futur/dx -------- #
        x_futur = u.x + u * timestep
        dx = u.x.diff('longitude')
        dxdx = x_futur.diff('longitude')/dx
        dxdx['longitude'] = eigengrid.longitude

        # ------- Computing dx_futur/dy -------- #
        # When derivating with respect to y we use v coords in the Arakawa C grid
        x_futur = v.x + u.interp(latitude=v.latitude, longitude=v.longitude, kwargs={'fill_value': None}) * timestep #fill_value=None extrapolates
        dy = v.y.diff('latitude')
        dxdy = x_futur.diff('latitude')/dy
        dxdy['latitude'] = eigengrid.latitude

        # ------- Computing dy_futur/dy -------- #
        y_futur = v.y + v * timestep
        dy = v.y.diff('latitude')
        dydy = y_futur.diff('latitude')/dy
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


def interpolate_c_stagger(
        u: xr.DataArray,
        v: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Function to interpolate regular xr.DataArrays onto the Arakawa C stagger.

    :param u: zonal wind
    :param v: meridional wind
    :return: u, v interpolated in the Arakawa C stagger and h_grid in the midpoints
    """
    regular_u, delta_lat_u, delta_lon_u = _assert_regular_latlon_grid(u)
    regular_v, delta_lat_v, delta_lon_v = _assert_regular_latlon_grid(v)
    assert regular_u, 'u array lat lon grid is not regular'
    assert regular_v, 'v array lat lon grid is not regular'
    assert delta_lat_u == delta_lat_v, 'u and v lat are not compatible'
    assert delta_lon_u == delta_lon_v, ' u and v lon are not compatible'
    u_new = u.copy(data=np.zeros(u.shape))
    v_new = v.copy(data=np.zeros(v.shape))
    u_new['latitude'].values = u.latitude - delta_lat_u*0.5
    v_new['longitude'].values = v.longitude - delta_lon_v*0.5
    u_new = u_new.isel(latitude=slice(None, -1))
    v_new = v_new.isel(longitude=slice(None, -1))
    u_new = u.interp_like(u_new)
    v_new = v.interp_like(v_new)
    h_grid = xr.DataArray(0, dims=['latitude', 'longitude'], coords=[
        u_new.latitude, v_new.longitude
    ])
    return u_new, v_new, h_grid


def _assert_regular_latlon_grid(array: xr.DataArray) -> Tuple[bool, np.array, np.array]:
    """
    Method to assert if an array latitude and longitude dimensions are regular.
    :param array: xr.DataArray to be asserted
    :return: Tuple[bool, np.array, np.array]
    """
    delta_lat = (array.latitude.shift(latitude=1) - array.latitude).dropna('latitude').values
    delta_lat = np.unique(np.round(delta_lat, 5)) # TODO rounding because numpy unique seems to have truncation error
    delta_lon = (array.longitude.shift(longitude=1) - array.longitude).dropna('longitude').values
    delta_lon = np.unique(np.round(delta_lon, 5))
    if delta_lat.shape[0] == 1 and delta_lon.shape[0] == 1:
        regular = True
    else:
        regular = False
    return regular, delta_lat, delta_lon


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
        f, ax = plt.subplots(1,1,figsize=(14,14),subplot_kw={'projection': ccrs.PlateCarree()} )
        eigenarray.sel(time=time).plot.contourf(cmap='nipy_spectral', vmax=0.7, ax=ax, levels = 50)
        ax.coastlines(color='white')
        plt.savefig(f'tempfigs/eigenvalues_{time}.png')
        plt.close()

    for time in eigenarray.time.values:
        f, axarr = plt.subplots(1, 3, figsize=(3*14, 14), subplot_kw={'projection': ccrs.PlateCarree()})
        max_array, diff_array, labeled_array = find_maxima(eigenarray.sel(time=time))

        max_array.plot(ax=axarr[0], vmax=1, cmap='nipy_spectral')
        diff_array.plot(ax=axarr[1])
        labeled_array.where(labeled_array !=0).plot(ax=axarr[2], cmap='tab20')
        axarr[0].coastlines(color='white')
        axarr[1].coastlines()
        axarr[2].coastlines()

        plt.savefig(f'tempfigs/labeled_{time}.png')
        plt.close()
    print(None)