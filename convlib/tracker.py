import xarray as xr
from skimage.feature import blob_dog, blob_log, blob_doh

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


class Normalizer():
    '''
    Normalizer for xarray datasets
    '''

    def __init__(self, alongwith):
        self._alongwith = alongwith

    def fit(self, X, y=None):
        '''
        :ds: xarray  dataset
        :alongwith: list of sample dimensions
        '''
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')

        X = X.stack({'alongwith': self._alongwith})
        self._mean = X.mean('alongwith')
        self._stdv = X.var('alongwith')**0.5
        return self

    def transform(self, X):
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
        X = X.stack({'alongwith': self._alongwith})
        X = (X - self._mean)/self._stdv
        return X.unstack('alongwith')

    def inverse_transform(self, X):
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
        X = X.stack({'alongwith': self._alongwith})
        X = X * self._stdv + self._mean

        return X.unstack('alongwith')
