import xarray as xr
from typing import List
from warnings import warn

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
    for i, time in list(ds[seq_dim].values):
        idx = int(i/seq_len)
        mask.append(idx)

    ds['seq'] = ((seq_dim), mask)
    return ds