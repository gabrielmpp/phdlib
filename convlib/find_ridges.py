
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
import glob
import xarray as xr
import numpy as np

experiment_path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/experiment_timelen_8_e84311e4-2dce-43a8-9dbf-cf8c4d3bbcc9'
files = [f for f in glob.glob(experiment_path + "**/*partial_0*.nc", recursive=True)]

for idx, file in enumerate(files):
    print(f'Doing file {idx + 1} of {len(files)}')
    with open(experiment_path + '/config.txt') as f:
        config = eval(f.read())
    days = config['lcs_time_len'] / 4
    print(file)
    da = xr.open_dataarray(file)
    da = np.log(da) / days
    ridges = find_ridges_spherical_hessian(da, scheme='first_order', sigma=1)
    ridges = ridges.where(ridges < -2e-10, 0)
    ridges = ridges.where(ridges >= -2e-10, 1)
    ridges = filter_ridges(ridges, ftle=da, criteria=['mean_intensity', 'area', 'eccentricity'],
                           thresholds=[.6, 30, 0.9])
    ridges = ridges.where(ridges == 1)
    ridges.to_netcdf(experiment_path + '/partial_ridges_{0:03d}.nc'.format(idx))
    da.close()
