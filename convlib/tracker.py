import xarray as xr
from skimage.feature import blob_dog, blob_log, blob_doh
from convlib.xr_tools import read_nc_files
from skimage import measure
from skimage import filters


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
    threshold = array.quantile(0.7)
    array = array.where(array > threshold, 0)
    array = array.where(array < threshold, 1)
    #labeled_array = xr.apply_ufunc(lambda x: measure.label(x), array)
    labeled_array = xr.apply_ufunc(lambda x: measure.label(x),
                                   array.stack(points=['latitude', 'longitude']).groupby('points'))
    labeled_array