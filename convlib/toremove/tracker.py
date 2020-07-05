import xarray as xr
from skimage.feature import blob_dog, blob_log, blob_doh
from convlib.xr_tools import read_nc_files
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
class Tracker:

    def __init__(self):
        from skimage.feature import blob_dog, blob_log, blob_doh
        pass

    def _identify_features(self, array):
        array = (array - array.min()) / array.max()
        for time in array.time.values:
            array2D = array.sel(time=time).values
            array2D.shape()
            blobs_doh = blob_doh(array2D, max_sigma=30, threshold=.01)
            print(';a')


        pass

    def track(self, array):
        identified_array = self._identify_features(array)


if __name__ == '__main__':
    array = read_nc_files(year_range=range(2000, 2001))
    array = array.sel(latitude=slice(-30,None), longitude=slice(-60, -30),time=slice(None,10))
    threshold = array.quantile(0.7)
    array = array.where(array > threshold, 0)
    array = array.where(array < threshold, 1)

    #labeled_array = xr.apply_ufunc(lambda x: measure.label(x), array)
    labeled_array = xr.apply_ufunc(lambda x: measure.label(x),
                                   array.groupby('time'))
    array = array.expand_dims(groups=np.unique(labeled_array.values))

    array_of_sizes = np.zeros([len(array.time.values),len(array.groups.values)])
    list_of_sizes = []
    for k, time in enumerate(labeled_array.time.values):
        print(f'{time}')
        temp = labeled_array.sel(time=time)
        array_temp = array.sel(time=time)
        for i, group in enumerate(array.groups.values):
            array_temp = array_temp.where(temp != group, array_temp.where(temp == group).sum().values)
        list_of_sizes.append(array_temp)
    """
    array.to_netcdf('/home/users/gmpp/temp_array.nc')
    array.coords['sizes'] = ('time','groups'), array_of_sizes
    array_big = array.where(array['sizes'] > 1000, 0)
    array['sizes'].resample(time='5D').mean('time').mean('groups').plot()
    plt.savefig('tempfigs/analysis/sizes_.png')
    array_big.isel(groups=1, time=1).plot()
    plt.savefig('tempfigs/analysis/big_.png')
    """