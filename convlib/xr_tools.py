import xarray as xr
from typing import List
from warnings import warn
import numpy as np
import traceback
import cmath
import pandas as pd

def createDomains(region, reverseLat=False):
    if region == "SACZ_big":
        domain = dict(latitude=[-40, 5], longitude=[-70, -20])
    elif region == "SACZ":
        domain = dict(latitude=[-40, -5], longitude=[-62, -20])
    elif region == "SACZ_small":
        domain = dict(latitude=[-30, -20], longitude=[-50, -35])
    elif region == "AITCZ":
        domain = dict(latitude=[-5, 15], longitude=[-45, -1])
    elif region == "NEBR":
        # domain = dict(latitude=[-15, 5], longitude=[-45, -15])
        domain = dict(latitude=[-10, 5], longitude=[-55, -40])
    elif region is None:
        domain = dict(latitude=[None, None], longitude=[None, None])
    else:
        raise ValueError(f'Region {region} not supported')

    if reverseLat:
        domain = dict(latitude=slice(domain['latitude'][1], domain['latitude'][0]),
                      longitude=slice(domain['longitude'][0], domain['longitude'][1]))

    else:
        domain = dict(latitude=slice(domain['latitude'][0], domain['latitude'][1]),
                      longitude=slice(domain['longitude'][0], domain['longitude'][1]))

    return domain


def read_nc_files(region=None,
                  basepath="/group_workspaces/jasmin4/upscale/gmpp/convzones/",
                  filename="SL_repelling_{year}_lcstimelen_1.nc",
                  year_range=range(2000, 2008), transformLon=False, lonName="longitude", reverseLat=False,
                  time_slice_for_each_year=slice(None, None), season=None, lcstimelen=None, set_date=False,
                  binary_mask=None, maskdims={'latitude': 'lat', 'longitude': 'lon'}):
    """

    :param binary_mask:
    :param transformLon:
    :param lonName:
    :param region:
    :param basepath:
    :param filename:
    :param year_range:
    :return:
    """
    print("*---- Starting reading data ----*")
    years = year_range
    file_list = []

    def transform(x, binary_mask):
        if transformLon:
            x.coords[lonName].values = \
                (x.coords[lonName].values + 180) % 360 - 180
        if not isinstance(region, type(None)):
            x = x.sel(createDomains(region, reverseLat))
        if not isinstance(season, type(None)):
            if set_date:
                initial_date = pd.Timestamp(f'{year}-01-01T00:00:00') + pd.Timedelta(str(lcstimelen*6 - 6) + 'H')
                final_date = pd.Timestamp(f'{year}-12-31T18:00:00')
                freq = '6H'
                x = x.assign_coords(time=pd.date_range(initial_date, final_date, freq=freq))
            if season == 'DJF':
                season_idxs = np.array([pd.to_datetime(t).month in [1, 2, 12] for t in x.time.values])
            elif season == 'JJA':
                season_idxs = np.array([pd.to_datetime(t).month in [5, 6, 7] for t in x.time.values])
            else:
                raise ValueError(f"Season {season} not supported")
            x = x.sel(time=x.time[season_idxs])
        if isinstance(binary_mask, xr.DataArray):
            binary_mask = binary_mask.where(binary_mask == 1, drop=True)
            x = x.sel(latitude=binary_mask[maskdims['latitude']].values,
                      longitude=binary_mask[maskdims['longitude']].values, method='nearest')
            binary_mask = binary_mask.rename({maskdims['longitude']: 'longitude', maskdims['latitude']: 'latitude'})
            x = x.where(binary_mask == 1, drop=True)

        return x

    for i, year in enumerate(years):
        print(f'Reading year {year}')
        filename_formatted = basepath + filename.format(year=year)
        print(filename_formatted)
        year = str(year)
        array = None
        fs = (xr.open_dataarray, xr.open_dataset)
        for f in fs:
            try:
                array = f(filename_formatted)
            except ValueError:
                print('Could not open file using {}'.format(f.__name__))
            else:
                break

        if isinstance(array, (xr.DataArray, xr.Dataset)):
            file_list.append(transform(array, binary_mask).isel(time=time_slice_for_each_year))
        else:
            print(f'Year {year} unavailable')
    print(file_list)
    full_array = xr.concat(file_list, dim='time')
    print('*---- Finished reading data ----*')
    return full_array


# def get_xr_seq(ds: xr.DataArray, seq_dim: str, idx_seq: List[int]):
#    """
#    Function that create the sequence dimension in overlapping time intervals
#
#    :param ds:
#    :param seq_dim:
#    :param idx_seq:
#    :return: xr.DataArray
#    """
#    dss = []
#    for idx in idx_seq:
#        dss.append(ds.shift({seq_dim: -idx}))

#    dss = xr.concat(dss, dim='seq')
#    dss = dss.assign_coords(seq=idx_seq)

#    return dss


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


def to_cartesian(array, lon_name='longitude', lat_name='latitude', earth_r=6371000):
    """
    Method to include cartesian coordinates in a lat lon xr. DataArray

    :param array: input xr.DataArray
    :param lon_name: name of the longitude dimension in the array
    :param lat_name: name of the latitude dimension in the array
    :param earth_r: earth radius
    :return: xr.DataArray with x and y cartesian coordinates
    """
    array['x'] = array[lon_name] * np.pi * earth_r / 180
    array['y'] = xr.apply_ufunc(lambda x: np.sin(np.pi * x / 180) * earth_r, array[lat_name])
    return array


def xy_to_latlon(x, y, earth_r=6371000):
    """
    Inverse function of meteomath.to_cartesian
    """
    longitude = x * 180 / (np.pi * earth_r)
    latitude = np.arcsin(y / earth_r) * 180 / np.pi
    return latitude, longitude


def get_xr_seq(ds, commun_sample_dim, idx_seq):
    """
    Internal function that create the sequence dimension
    :param ds:
    :param commun_sample_dim:
    :param idx_seq:
    :return:
    """
    dss = []
    for idx in idx_seq:
        dss.append(ds.shift({commun_sample_dim: -idx}))

    dss = xr.concat(dss, dim='seq')
    dss = dss.assign_coords(seq=idx_seq)

    return dss
