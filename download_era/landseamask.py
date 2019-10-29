# Import cdsapi
'''
Vertical integral of eastward water vapour flux
'''

import cdsapi
import sys
if __name__ == '__main__':
    # Open a new Client instance
    c = cdsapi.Client()
    outpath = '/gws/nopw/j04/primavera1/observations/ERA5/'
    # Send your request (download data)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': 'land_sea_mask',
            'year': '1980',
            'month': '07',
            'day': '01',
            'time': '00:00'
        },outpath+'landSea_mask.nc')
