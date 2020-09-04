
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
import glob
import xarray as xr
import numpy as np


experiment_path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/' \
                  'experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4'
files = [f for f in glob.glob(experiment_path + "**/*partial_0*.nc", recursive=True)]

for idx, file in enumerate(files):
    print(f'Doing file {idx + 1} of {len(files)}')
    with open(experiment_path + '/config.txt') as f:
        config = eval(f.read())
    days = config['lcs_time_len'] / 4
    print(file)
    da = xr.open_dataarray(file)
    da = np.log(da) / days
    ridges_l = []
    for time in da.time.values:
        da_ = da.sel(time=time)
        ridges, _ = find_ridges_spherical_hessian(da_,
                                                  sigma=1,
                                                  scheme='second_order',
                                                  angle=20)
        print(ridges)
        print(da_)
        da_ = da_.interp(latitude=ridges.latitude, longitude=ridges.longitude)

        ridges = filter_ridges(ridges, da_, criteria=['mean_intensity', 'major_axis_length'],
                               thresholds=[1.2, 30])

        ridges_l.append(ridges)
    ridges = xr.concat(ridges_l, dim='time')
    ridges.to_netcdf(experiment_path + '/partial_ridges_{0:03d}.nc'.format(idx))
    da.close()
