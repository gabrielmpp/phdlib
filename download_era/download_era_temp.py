# Import cdsapi
import cdsapi
import sys
if __name__ == '__main__':
    # Open a new Client instance
    c = cdsapi.Client()
    year = str(sys.argv[1])
    outpath = '/gws/nopw/j04/primavera1/observations/ERA5/'
    #outpath = '/group_workspaces/jasmin4/upscale/ERA5/'
    # Send your request (download data)
    c.retrieve('reanalysis-era5-pressure-levels', {
            'product_type': 'reanalysis',
            'param': '130',
            'year':  year,
            "pressure_level": ["925"],
            "month":["01","02","03","04","05","06","07","08","09","10","11","12"],
            "day":  ["01","02","03","04","05","06","07","08","09","10","11",
                                    "12","13","14","15","16","17","18","19","20","21","22",
                                    "23","24","25","26","27","28","29","30","31"],
            'time': ['00','06','12','18'],'format': 'netcdf',
        }, outpath+'t_925_ERA5_6hrPlevPt_{year}010100-{year}123118.nc'.format(year=year))
