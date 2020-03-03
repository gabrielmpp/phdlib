"""
 THE JUS MAKER !
non-linear dimensionality reduction in a xarray

"""

import xarray as xr
import umap
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import warnings

from PyEMD import EMD  # Empirical mode decomposition

# from dask import compute, delayed
from joblib import Parallel, delayed

from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_multiindex_name(array):
    """
    Method to check which dimensions are Multiindex

    :param array: xr.DataArray
    :return: list of multiindex dimensions
    """
    level_coords = getattr(array, "_level_coords", None)
    if not isinstance(level_coords, type(None)):
        multiindex = np.unique(list(level_coords.values())).tolist()
        stacked_dims = list(level_coords.keys())
    else:
        multiindex = []
        stacked_dims = []

    return multiindex, stacked_dims



def load_xru(path):
    file = open(path, 'rb')
    xru = pickle.load(file)

    return xru


class DummyTransformer:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X


class autoencoder():
    '''

    Multidimensional prepocessing

    Available modes:

    -UMAP
    Class to train and run a non-linear dimensionality reduction in a
    xarray.Dataset.  The encoding is performed thorugh uniform manifold
    projection (UMAP, https://github.com/lmcinnes/umap) and the decoding
    is performed with a 2-layer fully-connected network.

    -PCA
    -EMD
    -normalizer

    '''

    def __init__(
            self, dims_to_reduce=None, alongwith=None, n_components=None, n_neighbors=None,
            windows=False, window_size=None, window_dims=None, parallel=False, mode='umap'):
        '''
        Initializing the UMAP instance. TODO: Include all umap parameters
        '''
        assert isinstance(alongwith, (list, type(None))), "alongwith must be a list of strings"
        assert isinstance(window_dims, (list, type(None))), "window_dims must be a list of strings"
        assert isinstance(window_size, (int, type(None))), "window_size must be int"
        assert isinstance(parallel, bool), "parallel must be boolean"
        assert isinstance(n_components, (int, type(None)))
        assert isinstance(n_neighbors, (int, type(None)))

        if isinstance(dims_to_reduce, type(None)):
            warnings.warn('If dims to reduce is None, a dummy dimension will be created.')
        else:
            assert isinstance(dims_to_reduce, (list, type(None))), "dims_to_reduce must be a list of strings or None"

        self.n_components = n_components

        reducer_dict = dict(
            umap=umap.UMAP(n_components=n_components, n_neighbors=n_neighbors),
            pca=PCA(n_components),
            emd=sk_emd(n_components),
            standardscaler=StandardScaler(),
            MinMaxScaler=MinMaxScaler(feature_range=(0, 1)),
            quantile_transform=QuantileTransformer(),
            dummy_transformer=DummyTransformer()  # for checking purposes
        )
        if mode in reducer_dict.keys():
            self.reducer = reducer_dict[mode]
        else:
            raise NotImplementedError("Mode not available.")

        self.mode = mode
        self.n_neighbors = n_neighbors
        self.dims_to_reduce = dims_to_reduce
        self.alongwith = alongwith
        self.windows = windows
        self.window_size = window_size
        self.window_dims = window_dims
        self.parallel = parallel

    def save(self, path):
        file = open(path, 'wb')
        pickle.dump(self, file)  # save current object

    def _xarray_to_2D(self, array):

        array = array.stack({'dim_to_reduce': self.dims_to_reduce})
        array = array.stack({'alongwith': self.alongwith})

        original_shape = array.shape
        array = array.dropna(dim='alongwith', how='any')
        # array = array.dropna(dim='dim_to_reduce', how='any')

        if original_shape != array.shape:
            print('=' * 20)
            warnings.warn("Warning XRU: dropna on dim alongwidth removed data. "
                          "Original array shape is {} -> current shape {}".format(str(original_shape),
                                                                                  str(array.shape)))

        array = array.transpose('alongwith', 'dim_to_reduce')
        # array = array.transpose('alongwith', 'dim_to_reduce','dim_to_reduce')
        return array

    def plot_embedding(self, path, cmap='rainbow'):
        if not isinstance(self.encoded, type(None)):
            x = self.encoded.encoded_dims.values
            y = self.encoded.alongwith.values
            z = self.encoded.values
            plt.contourf([x, y], z, cmap=cmap)
            plt.savefig(path)
            plt.close()
        else:
            raise Exception("Encode the array first")

    def _get_idxs_windows(self, X, window_size, window_dims):
        """
        Method to retrieve the indices of fixed sized windows (N-D cube)
        Boundary condition: remove outside values
        """
        idxs_l = [np.arange(len(X[coord])) for coord in
                  window_dims]  # list of arrays containing the indices of each coord in the window dim
        coord_l = [X[coord] for coord in window_dims]  # list of the actual coord values
        grid_idxs = np.meshgrid(*tuple(idxs_l))  # grid of coord indexes [[0,1,2...],[0,1,2...]
        grid_coords = np.meshgrid(*tuple(coord_l))  # grid of coord values [[14.5,12.1,....],[1.3,3.5,...]]
        idxs_list = []
        wdw_centers = []
        grid_idxs = [g.T for g in grid_idxs]  # what a beauty!
        grid_coords = [g.T for g in grid_coords]
        for i, j in zip(np.nditer(grid_idxs), np.nditer(grid_coords)):
            idx = []
            wdw_coords = {}
            for idim, x in enumerate(list(i)):
                idx.append(self._window_idxs(x, grid_idxs, idim, window_size))
                wdw_coords[window_dims[idim]] = j[idim]
            idxs_list.append(idx)
            wdw_centers.append(wdw_coords)
        return idxs_list, wdw_centers

    def _window_idxs(self, x, grid, idim, window_size):
        grid_shape = np.array(grid).shape
        idxs = np.arange(x - window_size, x + window_size)
        idxs = idxs[idxs >= 0]
        idxs = idxs[idxs <= (grid_shape[idim + 1] - 1)]
        return idxs.tolist()

    # def _run_windows(self, i, model_id, X)
    # TODO: define function to call with dask
    def _fit_windows(self, X, window_size, window_dims):

        print(" --- Training UMAP models for each point in window_dims ---")
        idxs_list, wdw_centers = self._get_idxs_windows(X, window_size, window_dims)
        self.model_ids = range(len(wdw_centers))
        self.idxs_list = idxs_list
        xru_models = {}
        cropped_data = {}
        for i, model_id in enumerate(self.model_ids):
            cropped_data[model_id] = self._xarray_to_2D(
                X.isel(dict([(dim, idxs_list[i][j]) for j, dim in enumerate(window_dims)])))

        if not self.parallel:
            for i, model_id in enumerate(self.model_ids):
                xru_models[str(model_id)] = self._run_fit_windows(deepcopy(self.reducer), cropped_data[model_id],
                                                                  i, wdw_centers, model_id)[1]
                del model
                sys.stdout.write('\r{prog}%'.format(prog=round(i * 100 / len(wdw_centers))))
                sys.stdout.flush()
            # self.xru_models = xru_models
        else:
            result = Parallel(n_jobs=6, prefer="threads")(
                delayed(self._run_fit_windows)(deepcopy(self.reducer), cropped_data[model_id], i,
                                               wdw_centers,
                                               model_id) for i, model_id in enumerate(self.model_ids))
            xru_models = dict(result)

        return xru_models

    @staticmethod
    def _run_fit_windows(model, cropped_data, i, wdw_centers, model_id):
        xru_model = model.fit(cropped_data.values)
        sys.stdout.write('\r{prog}%'.format(prog=round(i * 100 / len(wdw_centers))))
        sys.stdout.flush()
        return model_id, xru_model

    def _transform_windows(self, X, xru_models, inverse=False):
        encoded = []
        if inverse:
            transform_type = 'inverse_transform'
        else:
            transform_type = 'transform'
        for i, model_id in enumerate(self.model_ids):
            model = xru_models[model_id]
            cropped_data = X.isel(dict([(dim, self.idxs_list[i][j]) for j, dim in enumerate(self.window_dims)]))

            encoded.append(getattr(model, transform_type)(self._xarray_to_2D(cropped_data).values))
        X = X.stack({'model_ids': self.window_dims})
        X = X.stack({'alongwith': self.alongwith})
        X = X.dropna(dim='alongwith', how='all')

        encoded = xr.DataArray(encoded,
                               coords=[X['model_ids'], X['alongwith'], range(self.n_components)],
                               dims=['model_ids', 'alongwith', 'encoded_dims'])
        encoded = encoded.unstack('alongwith')
        encoded = encoded.unstack('model_ids')
        return encoded

    def _fit_dask(self, X_name, X_element):
        print("*" * 20 + "Fitting {} per dim_to_groupby".format(self.mode) + "*" * 20)
        reducer = deepcopy(self.reducer)
        X_element = X_element.drop('dim_to_groupby')
        if hasattr(X_element, 'dim_to_groupby'):  # TODO pull request xarray
            X_element = X_element.squeeze('dim_to_groupby')
        X_element = self._xarray_to_2D(X_element)
        # self.fitted_umaps[str(X_name[0])] = reducer.fit(X_element.values)
        return (str(X_name[0]), reducer.fit(X_element))

    def _transform_dask(self, X_name, X_element, inverse=False):
        """

        :param X_name:
        :param X_element:
        :return:
        """
        dim_to_groupby = X_element[
            'dim_to_groupby']  # saving element coord to concat in the right order after dask operation

        X_element = X_element.drop('dim_to_groupby')
        if hasattr(X_element, 'dim_to_groupby'):  # TODO pull request
            X_element = X_element.squeeze('dim_to_groupby')

        X_element = self._xarray_to_2D(X_element)

        encoded = self.fitted_umaps[str(X_name[0])].inverse_transform(X_element) if inverse else self.fitted_umaps[
            str(X_name[0])].transform(X_element)

        encoded = xr.DataArray(encoded,
                               coords=[X_element['alongwith'], range(encoded.shape[1])],
                               dims=['alongwith', 'encoded_dims'])
        try:  # TODO FIX IT ASAP
            encoded['dim_to_groupby'] = dim_to_groupby  # recovering dim_to_groupby
            encoded = encoded.expand_dims('dim_to_groupby')
        except ValueError:
            encoded = encoded.expand_dims('dim_to_groupby')
            encoded['dim_to_groupby'] = dim_to_groupby  # recovering dim_to_groupby

        ds_encoded = encoded.unstack('alongwith')
        return ds_encoded

    def fit(self, X):
        '''
        Method to train a umap encoder

        :X: xarray dataset or dataarray
        :dims_to_reduce: list of dims to reduce
        :alongwith: str dim of samples (e.g. 'time')

        :return: trained umap model

        '''
        use_dummy = False  # by default do not use dummy

        multindex_name, stacked_dims = get_multiindex_name(X)
        if multindex_name:
            raise NotImplementedError(
                'Does not work with multi-index. Dimensions with multi-index in inputs are {}'.format(
                    ' '.join(multindex_name)))

        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
            self.dims_to_reduce.append('var')

        if isinstance(self.dims_to_reduce, type(None)):
            use_dummy = True
            X = X.expand_dims('dummy')
            self.dims_to_reduce = ['dummy']

        self.original_dims = X.dims  # Tom: useful to make sure the outputs array is in the correct order
        self.dims_to_groupby = list(X.dims)
        [self.dims_to_groupby.remove(dim) for dim in self.alongwith]
        [self.dims_to_groupby.remove(dim) for dim in self.dims_to_reduce]

        if not self.windows:

            if len(self.dims_to_groupby):
                X = X.stack({'dim_to_groupby': self.dims_to_groupby})
                X_list = list(X.groupby('dim_to_groupby'))
                self.fitted_umaps = {}
                if self.parallel:
                    values = [delayed(self._fit_dask)(X_name, X_element) for X_name, X_element in X_list]
                    dummy = compute(*values, scheduler='threads')
                    dummy = dict(dummy)
                else:
                    dummy = {}
                    for X_name, X_element in X_list:
                        name, fitted_reducer = self._fit_dask(X_name, X_element)
                        dummy[name] = fitted_reducer
                self.fitted_umaps = dummy

            else:
                X = self._xarray_to_2D(X)
                self.reducer.fit(X)

        else:
            if len(self.dims_to_groupby):
                X = X.stack({'dim_to_groupby': self.dims_to_groupby})
                X_list = list(X.groupby('dim_to_groupby'))
                self.fitted_umaps = {}
                for X_name, X_element in X_list:
                    fitted_reducer = self._fit_windows(X_element.isel(dim_to_groupby=0),
                                                       self.window_size, self.window_dims)
                    self.fitted_umaps[X_name] = fitted_reducer
            else:
                self.fitted_umaps = dict(xru_models=self._fit_windows(X, self.window_size, self.window_dims))

        print("done fit xru ;)")
        if use_dummy:
            self.dims_to_reduce = None  # Returning dims_to_reduce to None after adding the dummy dimension temporarily
        return self

    def transform(self, X, inverse=False):
        """
        Method to run the umap encoder on new data
        :X: xarray dataset or dataarray
        :dims_to_reduce: list of dims to reduce
        :alongwith: str dim of samples (e.g. 'time')

        :return: xarray.Dataset with encoded dimensions
        """

        if isinstance(self.dims_to_reduce, type(None)):
            X = X.expand_dims('dummy')
            self.dims_to_reduce = ['dummy']

        if inverse:
            inverse_transform = getattr(self.reducer, "inverse_transform", None)
            if not callable(inverse_transform):
                raise Exception("Mode does not support inverse transform")

        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')

        if not self.windows:
            if not len(self.dims_to_groupby):
                X = self._xarray_to_2D(X)
                # self._array = X # TODO TO BE FIXED/REMOVED
                encoded = self.reducer.inverse_transform(X) if inverse else self.reducer.transform(X)
                encoded = xr.DataArray(encoded,
                                       coords=[X['alongwith'], range(encoded.shape[1])],
                                       dims=['alongwith', 'encoded_dims'])

                ds_encoded = encoded.unstack('alongwith')
            else:

                X = X.stack({'dim_to_groupby': self.dims_to_groupby})

                X_list = list(X.groupby('dim_to_groupby'))
                self.encoded_arrays = []
                if self.parallel:
                    values = [delayed(self._transform_dask)(X_name, X_element, inverse=inverse) for X_name, X_element in
                              X_list]
                    result = list(compute(*values, scheduler='processes'))
                else:
                    result = []
                    for X_name, X_element in X_list:
                        result.append(self._transform_dask(X_name, X_element, inverse=inverse))
                ds_encoded = xr.concat(result, dim='dim_to_groupby')
                ds_encoded = ds_encoded.assign_coords(dim_to_groupby=X.dim_to_groupby)
                ds_encoded = ds_encoded.unstack()

                del self.encoded_arrays

                print('done')
        else:
            if len(self.dims_to_groupby):
                X = X.stack({'dim_to_groupby': self.dims_to_groupby})
                X_list = list(X.groupby('dim_to_groupby'))
                result = []
                for X_name, X_element in X_list:
                    result.append(
                        self._transform_windows(
                            X_element.isel(dim_to_groupby=0),
                            self.fitted_umaps[X_name], inverse=inverse)
                    )
                ds_encoded = xr.concat(result, dim='dim_to_groupby')
                ds_encoded = ds_encoded.assign_coords(dim_to_groupby=X.dim_to_groupby)
                ds_encoded = ds_encoded.unstack()
            else:
                ds_encoded = self._transform_windows(X, self.fitted_umaps)

        # ds_encoded = ds_encoded.transpose(self.original_dims) # TODO THIS MUST WORK!
        return ds_encoded

    def train_decoder(self, x=None, y=None):
        '''
        Method to train the decoder neural network
        :x: encoded dataset
        :y: original dataset

        :return: xarray.Dataset with encoded dimensions
        '''

        if isinstance(x, type(None)):
            x = self.encoded
        else:
            raise Exception('Not implemented yet')
        #
        # if isinstance(y, type(None)): # TODO TO BE FIXED
        #     y = self._array
        # else:
        #     self._array = y

        self._decoder_nn(x.values, y.values)
        return self

    def _decoder_nn(self, x, y, out_activation='linear'):
        '''
        Vanilla configuration for the decoder. TODO: allow user define
        architecture.
        '''

        epochs = 30
        main_input = Input(shape=x.shape[1:], name="main_input")
        dense1 = Dense(
            100, bias_initializer="zeros", use_bias=True, activation="relu")(main_input)
        dense2 = Dense(
            100, bias_initializer="zeros", use_bias=True, activation="relu")(dense1)
        out = Dense(y.shape[1], activation=out_activation)(dense2)
        model = Model(main_input, out)
        model.compile(
            loss="mean_squared_error",
            metrics=[metrics.mae, metrics.mse],
            optimizer="adam",
        )
        history = model.fit(
            x,
            y,
            epochs=epochs,
            validation_split=0.2,
            shuffle=True,
            verbose=2,
            batch_size=28,
        )
        self.decoder = model

    def decode(self, x=None):
        '''
        Method to run the decoder NN on new data
        :x: xarray dataset or dataarray
        '''
        if not isinstance(self.decoder, type(None)):
            if isinstance(x, type(None)):
                x = self.encoded

            decoded = self.decoder.predict(x.values)

            # new_ds = xr.DataArray(decoded,  # TODO TO BE FIXED
            #                     coords=self._array.coords,
            #                     dims=self._array.dims)

            self.decoded = new_ds.unstack(['dim_to_reduce', 'alongwith'])
            return self.decoded

        else:
            raise Exception("Run train_decoder first")


class normalizer():
    '''
    Normalizer for xarray datasets
    '''

    def __init__(self, alongwith):
        self._alongwith = alongwith

    def fit(self, X):
        '''
        :ds: xarray  dataset
        :alongwith: list of sample dimensions
        '''
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')

        X = X.stack({'alongwith_of_norm': self._alongwith})  # TODO note: xru.encoder already passed an alongwith
        self._mean = X.mean('alongwith_of_norm')
        self._stdv = X.var('alongwith_of_norm') ** 0.5
        return self

    def transform(self, X):
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
        X = X.stack({'alongwith_of_norm': self._alongwith})
        X = (X - self._mean) / self._stdv
        return X.unstack('alongwith_of_norm')

    def inverse_transform(self, X):
        if isinstance(X, xr.Dataset):
            X = X.to_array(dim='var')
        X = X.stack({'alongwith_of_norm': self._alongwith})
        X = X * self._stdv + self._mean
        return X.unstack('alongwith_of_norm')


class sk_emd():
    '''

    Wrapper between PyEMD and sklearn
    Empirical Mode Decomposition for xarray
    '''

    def __init__(self, nb_components=None):
        self.nb_components = nb_components  # TODO Note: EMD return a variable maximum number of components generally between 12 and 14

    def fit(self, dummy_input=None):
        '''
        :ds: xarray  dataset
        :alongwith: list of sample dimensions
        '''
        emd = EMD()
        self.emd = emd

        return self

    def transform(self, ds_x, return_2d=True):
        """
        Return decomposed signal

        :param ds_x: dataarray
        : return_2d: bool, if True return the result as a 2d array
        :return:
        """

        if len(ds_x.shape) is not 2:
            # TODO should work with 1D
            raise ('sk_emd only work with 2d array (sample, features)')  # TODO implement multidimenstional sk_emd

        dss_ifs = []
        for feature_idx in range(ds_x.shape[1]):
            print('EMD decomposition of feature {}'.format(str(feature_idx)))
            s = ds_x.values[:, feature_idx]
            model_emd = deepcopy(self.emd)
            model_emd(s)
            imfs, residual = model_emd.get_imfs_and_residue()

            if self.nb_components is not None:
                imfs_lowfreq = imfs[-self.nb_components:, :]  # low freq
                imfs_highfreq = imfs[self.nb_components:, :]
                residual = np.vstack([imfs_highfreq, residual])
                imfs = imfs_lowfreq
                residual = residual.sum(axis=0)

                print('There are {} modes selected '.format(imfs.shape[0]))

            imfs = np.vstack([imfs, residual])  # add residual

            ds_ifs = xr.DataArray(imfs, dims=['imfs', ds_x.dims[0]],
                                  coords={'imfs': range(imfs.shape[0]), ds_x.dims[0]: ds_x.coords[ds_x.dims[0]]})

            dss_ifs.append(ds_ifs)
        dss_ifs = xr.concat(dss_ifs, dim=ds_x.dims[1])
        dss_ifs[ds_x.dims[1]] = ds_x.coords[ds_x.dims[1]]

        if return_2d:
            dss_ifs = dss_ifs.rename({ds_x.dims[1]: 'features'})
            dss_ifs = dss_ifs.reset_index(
                'features')  # TODO I reset the index to be able to make the stack on multindex. Find a more elegant way
            dss_ifs.coords['features'] = list(range(len(dss_ifs.features)))
            dss_ifs = dss_ifs.stack({ds_x.dims[1]: ['features', 'imfs']})  # TODO Does not work with multiindex

        return dss_ifs


if __name__ == '__main__':
    input_array = xr.open_dataarray('/path/to/ncdf')
    normalizer = xr.normalizer(alongwith=['time'])
    normalizer = normalizer.fit(input_array)
    input_array = normalizer.transform(input_array)

    # TODO Tom: on xrk I get an error when I use only one (e.g. ['lat']) xrk_grid dimension
    # The error appears at _get_idxs_windows
    # Maybe we should make a common external "get_idx_windows"
    # I added at line 320 on xrk
    # i = np.atleast_1d(i)
    # j = np.atleast_1d(j)

    model = autoencoder(n_components=100, n_neighbors=2,
                        dims_to_reduce=['lat', 'lon'], alongwith=['time'])
    model = model.fit(input_array)
    encoded_array = model.transform(input_array)
    model = model.train_decoder(encoded_array)
    decoded_array = model.decode()
