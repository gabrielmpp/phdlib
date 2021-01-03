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
    c.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'param': '137',
            'year':  year,
            "month":["01","02","03","04","05","06","07","08","09","10","11","12"],
            "day":  ["01","02","03","04","05","06","07","08","09","10","11",
                                    "12","13","14","15","16","17","18","19","20","21","22",
                                    "23","24","25","26","27","28","29","30","31"],
            'time': ['00', '03', '06', '12', '15', '18', '21'],'format': 'netcdf',
        }, outpath+'tcwv_ERA5_6hr_{year}010100-{year}123118.nc'.format(year=year))
