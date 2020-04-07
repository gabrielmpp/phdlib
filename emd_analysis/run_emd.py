from mia_lib.miacore import xrumap as xru
from pathlib import Path
import xarray as xr
import os


emd_kwargs = dict(  # DEFAULT EMD KWARGS - don't change here
    energy_ratio_thr=0.2,
    std_thr=0.2,
    svar_thr=0.001,
    total_power_thr=0.005,
    range_thr=0.001,
    MAX_ITERATION=1000,
)


relax_thresholds = 1  # relaxation factor > 1
max_imf = 9
for key in emd_kwargs.keys():
    emd_kwargs[key] = relax_thresholds * emd_kwargs[key]

xru_kwargs = dict(
alongwith=['time'],
    mode='ceemdan',
    parallel=True,
    nthreads=os.cpu_count() - 3,
    emd_kwargs=emd_kwargs
)
sp = {'lat': -23.5, 'lon': -46.5, 'method': 'nearest'}
cpc_path = Path('/home/users/gmpp/phd_data/precip_1979a2017_CPC_AS.nc')
da = xr.open_dataarray(cpc_path)
da = da.assign_coords(lon=(da.coords['lon'].values + 180) % 360 - 180)
ds = da.sel(time=slice('2008-01-01', None)).sel(**sp)

# MAX_ITERATION WILL CONTROL THE NUMBER OF SIFTING ITERATIONS
# THE OTHER THRESHOLDS CONTROL THE NUMBER OF EMD -


transformer_kwargs = dict()

transformer_kwargs['max_imf'] = max_imf

xru_transformer = xru.autoencoder(**xru_kwargs)
xru_transformer = xru_transformer.fit(ds, **xru_kwargs)
ds = xru_transformer.transform(ds, transformer_kwargs=transformer_kwargs, **xru_kwargs)
if ds.encoded_dims.shape[0] == 1:
    ds = ds.isel(encoded_dims=0).drop('encoded_dims')
print('saving')
ds.to_netcdf('/home/users/gmpp/ds_emd_sp.nc')
print('done')