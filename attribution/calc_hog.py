import xarray as xr
import glob
import matplotlib.pyplot as plt
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
import numpy as np
from dask.distributed import performance_report
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import dask
from skimage.feature import hog

outpath = '/home/gab/phd/data/hog_smoothed_spline/'
outpath_eigv = '/home/gab/phd/data/eigvectors/'
datapath = '/home/gab/phd/data/ridges_smoothed_spline/'
file_list = glob.glob(datapath + '*.nc')
file_list.sort()
da_list = []


def calc_hog(x):
    if 'time' in x.dims:
        x = x.isel(time=0)
    x = x.where(x == 1, 0)
    return x.copy(data=hog(x, visualize=True)[1])


def func_to_apply(x):
    ridge = x.groupby('time').apply(calc_hog)
    return ridge


for i, filename in enumerate(file_list):
    #  print('*--- Reading file {} of {} ---*'.format(i, filename))
    outname = filename.split('/')[-1]
    da = xr.open_dataarray(filename, chunks={'time': 60})
    da = da.sortby('time')
    # da = .5 * xr.apply_ufunc(np.log, da, dask='allowed')
    print(da)
    with ProgressBar():
        ridges = da.map_blocks(func_to_apply, template=da).compute(scheduler='processes', num_workers=10)

    print('Calculation end.')
    print('Writing output.')
    ridges.to_netcdf(outpath + 'hog_ridges_' + outname)
