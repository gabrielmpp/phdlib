import xarray as xr
import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
from meteomath import to_cartesian
from typing import Tuple
from meteomath import interpolate_c_stagger
from convlib.xr_tools import xy_to_latlon
from sklearn.preprocessing import MinMaxScaler


# import matplotlib.pyplot as plt

LCS_TYPES = ['attracting', 'repelling']


def double_gyre(x, y, t, freq=0.1, epsilon=0.25, nt=None):
    """
    Function to compute the double gyre analytically

    :param x: np.array with x positions
    :param y: np.array with y positions
    :param t: np.array with time values
    :param freq: temporal frequency
    :param epsilon: intensity of time dependence
    :return: tuple of numpy arrays with zonal and meridional components
    """

    omega = freq*2*np.pi
    try_to_make_full_turn = True

    if try_to_make_full_turn:
        A = 2*np.pi*0.5*(x.shape[0]+x.shape[1])/nt
    else:
        A = 0.1

    print(A)
    a = epsilon*np.sin(omega * t)
    b = 1 - 2*epsilon*np.sin(omega*t)
    f = a*x**2 + b*x
    u = -np.pi * A * np.sin(np.pi*f) * np.cos(np.pi*y)
    dfdx = 2*a*x + b
    v = np.pi * A * np.cos(np.pi*f)*np.sin(np.pi*y) * dfdx
    return u, v


class LCS:
    """
    Methods to compute LCS in 2D wind fields in xarrray dataarrays
    """

    def __init__(self, lcs_type: str, timestep: float = 1, dataarray_template=None, timedim='time', shearless=False):
        assert isinstance(lcs_type, str), "Parameter lcs_type expected to be str"
        assert lcs_type in LCS_TYPES, f"lcs_type {lcs_type} not available"
        self.lcs_type = lcs_type
        self.timestep = timestep
        self.timedim = timedim
        self.shearless = shearless

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
        timedim = self.timedim

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
        eigenvalues = eigenvalues.expand_dims({self.timedim: [u[self.timedim].values[0]]})


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
            [4])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
        eigenvalues.shape
        return eigenvalues

    @staticmethod
    def _parcel_propagation(u, v, timestep, propdim="time"):
        """
        Method to propagate the parcel given u and v

        :param u: xr.DataArray zonal wind
        :param v: xr.DataArray meridional wind
        :return: tuple of xr.DataArrays containing the final positions of the trajectories
        """

        times = u[propdim].values.tolist()
        if timestep < 0:
            times.reverse()  # inplace

        # initializing and integrating
        y_futur = u.y.values
        x_futur = u.x.values
        positions_x, positions_y = np.meshgrid(x_futur, y_futur)

        initial_pos = xr.DataArray()

        for time in times:
            print(f'Propagating time {time}')
            lat, lon = xy_to_latlon(y=positions_y, x=positions_x)
            lat = lat[:, 0]  # lat is constant along cols
            lon = lon[0, :]  # lon is contant along rows

            # ---- propagating positions ---- #

            y_buffer = positions_y + \
                       timestep * v.sel({propdim: time}).interp(latitude=lat.tolist(), method='linear',
                                                                longitude=lon.tolist(),
                                                                kwargs={'fill_value': 0}).values

            x_buffer = positions_x + \
                       timestep * u.sel({propdim: time}).interp(latitude=lat.tolist(), method='linear',
                                                                longitude=lon.tolist(),
                                                                kwargs={'fill_value': 0}).values
            # ---- Updating positions ---- #
            positions_x = x_buffer
            positions_y = y_buffer

        positions_x = xr.DataArray(positions_x, dims=['latitude', 'longitude'],
                                   coords=[u.latitude.values,u.longitude.values])
        positions_y = xr.DataArray(positions_y, dims=['latitude', 'longitude'],
                                   coords=[u.latitude.values,u.longitude.values])
        positions_x['x'] = (('longitude'), u.x.values)
        positions_x['y'] = (('latitude'), u.y.values)
        positions_y['x'] = (('longitude'), u.x.values)
        positions_y['y'] = (('latitude'), u.y.values)

        return positions_x, positions_y

    def _compute_deformation_tensor(self, u: xr.DataArray, v: xr.DataArray, timestep: float) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the meridional wind field
        :param timestep: float
        :return: xr.DataArray of the deformation tensor
        """

        x_futur, y_futur = self._parcel_propagation(u, v, timestep, propdim=self.timedim)
        # u, v, eigengrid = interpolate_c_stagger(u, v)
        dx = x_futur.x.diff('longitude')
        dy = y_futur.y.diff('latitude').values

        dxdx = x_futur.diff('longitude') / x_futur.x.diff('longitude')
        dxdy = x_futur.diff('latitude') / x_futur.y.diff('latitude')
        dydy = y_futur.diff('latitude') / y_futur.y.diff('latitude')
        dydx = y_futur.diff('longitude') / y_futur.x.diff('longitude')

        dxdx = dxdx.transpose('latitude', 'longitude')
        dxdy = dxdy.transpose('latitude', 'longitude').drop('x').drop('y')
        dydy = dydy.transpose('latitude', 'longitude').drop('x').drop('y')
        dydx = dydx.transpose('latitude', 'longitude').drop('y')
        if self.shearless:
            dydx = dydx*0
            dxdy = dxdy*0
        dxdx.name = 'dxdx'
        dxdy.name = 'dxdy'
        dydy.name = 'dydy'
        dydx.name = 'dydx'

        def_tensor = xr.merge([dxdx, dxdy, dydx, dydy])
        def_tensor = def_tensor.to_array()
        def_tensor = def_tensor.rename({'variable': 'derivatives'})
        def_tensor = def_tensor.transpose('derivatives', 'latitude', 'longitude')
        # from xrviz.dashboard import Dashboard
        # dashboard = Dashboard(def_tensor)
        # dashboard.show()
        print(def_tensor)
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
        dxdy = dxdy.transpose('latitude', 'longitude').drop('x') * 0
        dydy = dydy.transpose('latitude', 'longitude').drop('x').drop('y')
        dydx = dydx.transpose('latitude', 'longitude').drop('y') * 0
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

    nt = 200
    nx = 120
    ny = 60
    dx = 1
    epsilon = 0
    mode = 'repelling'
    backwards = True
    shearless = False
    latlon_array = xr.DataArray(np.zeros([ny, nx, nt]), dims=['latitude', 'longitude', 't'],
                         coords={'longitude': np.linspace(-80, -80+nx*dx, nx),
                                 'latitude': np.linspace(-60, -60+nx*dx, ny), 't': np.linspace(0, 1, nt)})
    latlon_array.isel(t=0).plot()
    plt.show()
    cartesian_array = to_cartesian(latlon_array)
    scalerx = MinMaxScaler(feature_range=(0,2))
    scalery = MinMaxScaler(feature_range=(0,1))

    scaled_x = scalerx.fit_transform(cartesian_array.x.values.reshape(-1,1))
    scaled_y = scalery.fit_transform(cartesian_array.y.values.reshape(-1,1))

    grid = np.meshgrid(scaled_x, scaled_y)
    u = []
    v = []
    for t in cartesian_array.t.values:
        v_temp, u_temp = double_gyre(grid[1], grid[0], t, epsilon, nt=cartesian_array.t.values.shape[0                                                                                              ])
        u.append(u_temp)
        v.append(v_temp)

    u = np.stack(u, axis=2)
    v = np.stack(v, axis=2)

    u = cartesian_array.copy(data=u)
    v = cartesian_array.copy(data=v)
    mag = (u**2 + v**2)**0.5
    mag.name = 'magnitude'

    #for time in range(nt):



    #    plt.streamplot(y=u.latitude.values,x=u.longitude.values, u=u.isel(t=time).values,
    #                       v=v.isel(t=time).values, color=mag.isel(t=time).values)

    #   plt.show()

    if backwards:
        lcs = LCS(mode, -1000, dataarray_template=None, timedim='t', shearless=shearless)
    else:
        lcs = LCS(mode, 1000, dataarray_template=None, timedim='t', shearless=shearless)

    eigenvalues = lcs(u=u, v=v)

    #eigenvalues.isel(t=0).plot()
    #plt.show()


    for time in range(nt):

        eigenvalues.isel(t=0).plot()
        plt.streamplot(x=u.longitude.values, y=u.latitude.values, u=u.isel(t=time).values,
                       v=v.isel(t=time).values, color=mag.isel(t=time).values)
        plt.show()
    print(u.max())
    from xrviz.dashboard import Dashboard
    dashboard = Dashboard(u)
    dashboard.show()
    plt.streamplot(x=u.longitude.values, y=u.latitude.values, u=u.isel(t=time).values,
                   v=v.isel(t=time).values, color=mag.isel(t=time).values)
    plt.show()