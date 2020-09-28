    # Import cdsapi
import cdsapi
import sys
import config

if __name__ == '__main__':
    # Open a new Client instance
    c = cdsapi.Client()
    year = str(sys.argv[1])
    outpath = config.outpath
    #outpath = '/group_workspaces/jasmin4/upscale/ERA5/'
    # Send your request (download data)

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'param': '129',
            'year': '1980',
            'month': '07',
            'day': '01',
            'time': '00:00'
        },outpath+'geopotential_orography.nc')
