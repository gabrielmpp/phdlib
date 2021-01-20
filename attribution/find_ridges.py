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

outpath_ridges = '/home/gab/phd/data/ridges_smoothed_spline/'
outpath_eigv = '/home/gab/phd/data/eigvectors_smoothed_spline/'
calc_eigv = True
if calc_eigv:
    outpath = outpath_eigv
    return_idx = -1
else:
    outpath = outpath_ridges
    return_idx = 0

datapath = '/home/gab/phd/data/ftle_smoothed_spline/'
file_list = glob.glob(datapath + '*.nc')
file_list.sort()
da_list = []


def find_ridges(x):
    if 'time' in x.dims:
        x=x.isel(time=0)
    return find_ridges_spherical_hessian(x, sigma=4, tolerance_threshold=0.0015e-3,
    scheme='second_order', return_eigvectors=True)[return_idx]


def func_to_apply(x):
    ridge = x.groupby('time').apply(find_ridges)
    ridge = ridge.where(x > 1.2)
    return ridge


for i, filename in enumerate(file_list[61:]):
    print(1)

    #  print('*--- Reading file {} of {} ---*'.format(i, filename))
    outname = filename.split('/')[-1]
    da = xr.open_dataarray(filename, chunks={'time': 60})
    da = da.sortby('time')
    da = .5 * xr.apply_ufunc(np.log, da, dask='allowed')
    print(da)
    with ProgressBar():
        ridges = da.map_blocks(func_to_apply, template=da).compute(scheduler='processes', num_workers=4)

    print('Calculation end.')
    print('Writing output.')
    ridges.to_netcdf(outpath + outname)
