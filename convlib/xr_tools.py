import xarray as xr
from typing import List


def get_xr_seq(ds: xr.DataArray, seq_dim: str,
               idx_seq: List[int]):  # TODO I took this from xrk, might put it on a xarray toolbox
    """
    Internal function that create the sequence dimension

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
