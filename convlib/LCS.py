
import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

class LCS:
    """
    Methods to compute LCS in 2D wind fields in xarrray dataarrays
    """

    def __init__(self):
        pass

    def __call__(self, u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:

        """
        :param u: xarray datarray containing u-wind component
        :param v: xarray dataarray containing v-wind component
        :return: xarray dataarray containg
        """

        def_tensor = self._compute_deformation_tensor(u, v)
        def_tensor = def_tensor.stack({'points': ['time', 'latitude', 'longitude']})
        def_tensor = def_tensor.dropna(dim='points')
        eigenvalues = xr.apply_ufunc(lambda x: self._compute_eigenvalues(x), def_tensor.groupby('points'))
        eigenvalues = eigenvalues.unstack('points')
        return eigenvalues

    @staticmethod
    def _compute_eigenvalues(def_tensor: np.array) -> np.array:
        d_matrix = def_tensor.reshape([2, 2])
        cauchy_green = np.matmul(d_matrix.T, d_matrix)
        eigenvalues = max(np.linalg.eig(cauchy_green.reshape([2, 2]))[
                              0])
        eigenvalues = np.repeat(eigenvalues, 4).reshape([4, 1])  # repeating the same value 4 times just to fill the xr.DataArray in a dummy dimension
        return eigenvalues

    @staticmethod
    def _compute_deformation_tensor(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
        """
        :param u: xr.DataArray of the zonal wind field
        :param v: xr.DataArray of the merifional wind field
        :return: xr.DataArray of the deformation tensor
        """
        timestep_u = pd.infer_freq(u.time.values)
        timestep_v = pd.infer_freq(v.time.values)
        assert timestep_u == timestep_v, "u and v timesteps are different!"
        timestep = timestep_u

        if 'H' in timestep:
            timestep = float(timestep.replace('H', '')) * 3600
        elif timestep == 'D':
            timestep = 86400
        else:
            raise ValueError(f"Frequence {timestep} not supported.")

        x_futur = u.x + u * timestep
        y_futur = u.y + v * timestep

        # Solid boundary conditions TODO: improve to more realistic case
        # x_futur = x_futur.where(x_futur > u.x.min(), u.x.min()).where(x_futur < u.x.max(), u.x.max())
        # y_futur = y_futur.where(y_futur > u.y.min(), u.y.min()).where(y_futur < u.y.max(), u.y.max())

        dxdx = x_futur.differentiate('x')
        dxdy = x_futur.differentiate('y')
        dydy = y_futur.differentiate('y')
        dydx = y_futur.differentiate('x')

        dxdx = dxdx.transpose('time', 'latitude', 'longitude')
        dxdy = dxdy.transpose('time', 'latitude', 'longitude')
        dydy = dydy.transpose('time', 'latitude', 'longitude')
        dydx = dydx.transpose('time', 'latitude', 'longitude')
        def_tensor = xr.concat([dxdx, dxdy, dydx, dydy],
                               dim=pd.Index(['dxdx', 'dxdy', 'dydx', 'dydy'], name='derivatives'))
        def_tensor = def_tensor.transpose('time', 'derivatives', 'latitude', 'longitude')
        def_tensor = def_tensor.isel(derivatives=0).drop('derivatives')
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