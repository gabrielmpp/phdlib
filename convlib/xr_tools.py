import xarray as xr
from typing import List
from warnings import warn
import numpy as np


domains = dict(
    AITCZ=dict(latitude=slice(-5, 15), longitude=slice(-50, -13)),
    SACZ=dict(latitude=slice(-40,-5), longitude=slice(-62,-20))
    )


def read_nc_files(region=None,
                  basepath="/group_workspaces/jasmin4/upscale/gmpp/convzones/",
                  filename="SL_repelling_{year}_lcstimelen_1.nc",
                  year_range=range(2000, 2008)):
    """

    :param region:
    :param basepath:
    :param filename:
    :param year_range:
    :return:
    """
    print("*---- Starting reading data ----*")
    years = year_range
    file_list = []
    for year in years:
        print(f'Reading year {year}')
        year = str(year)
        try:
            array = xr.open_dataarray(basepath + filename.format(year=year))

            if not isinstance(region, type(None)):
                array = array.sel(domains[region])

            file_list.append(array)
        except:
            print(f'Year {year} unavailable')

    full_array = xr.concat(file_list, dim='time')
    print('*---- Finished reading data ----*')
    return full_array


def get_xr_seq(ds: xr.DataArray, seq_dim: str, idx_seq: List[int]):
    """
    Function that create the sequence dimension in overlapping time intervals

    :param ds:
    :param seq_dim:
    :param idx_seq:
    :return: xr.DataArray
    """
    dss = []
    for idx in idx_seq:
        dss.append(ds.shift({seq_dim: -idx}))

    dss = xr.concat(dss, dim='seq')
    dss = dss.assign_coords(seq=idx_seq)

    return dss


def get_seq_mask(ds: xr.DataArray, seq_dim: str, seq_len: int):
    """
    Function that create the sequence dimension in non-overlapping time intervals

    :param ds:
    :param seq_dim:
    :param idx_seq:
    :return: xr.DataArray
    """
    mask = []
    quotient, remainder = divmod(len(ds[seq_dim].values), seq_len)
    assert quotient > 0, f'seq_len cannot be larger than {seq_dim} length!'

    if remainder != 0:
        warn(f"Length of dim {seq_dim} is not divisible by seq_len {seq_len}. Dropping last {remainder} entries.")
        ds = ds.isel({seq_dim: slice(None, len(ds[seq_dim].values) - remainder)})

    print(ds[seq_dim])
    for i, time in enumerate(ds[seq_dim].values.tolist()):
        idx = int(i / seq_len)
        mask.append(idx)

    ds['seq'] = ((seq_dim), mask)
    return ds


def xy_to_latlon(x, y, earth_r=6371000):
    """
    Inverse function of meteomath.to_cartesian
    """
    longitude = x * 180 / (np.pi * earth_r)
    latitude = np.arcsin(y / earth_r) * 180/np.pi
    return latitude, longitude
