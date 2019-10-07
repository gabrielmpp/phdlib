import xarray as xr

base_path = '/gws/nopw/j04/primavera1/observations/ERA5/'
files = dict(pres='sp_ERA5_6hr_{year}010100-{year}123118.nc',
             v='v_925_ERA5_6hrPlevPt_{year}010100-{year}123118.nc',
             t='t_700_ERA5_6hrPlevPt_{year}010100-{year}123118.nc')

domains = dict(
    AITCZ=dict(latitude=slice(-5, 15), longitude=slice(-50, -13)),
    SACZ=dict(latitude=slice(-40,-5), longitude=slice(-62,-20)))

def read_nc_files(region,
                  basepath,
                  filename,
                  year_range=range(1980, 2008)):
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
        array = xr.open_dataarray(basepath + filename.format(year=year))
        array = array.sel(domains[region])
        array = array.resample(time="1D").mean()
        file_list.append(array)
    full_array = xr.concat(file_list, dim='time')
    print('*---- Finished reading data ----*')
    return full_array

def main():
    region = "SACZ"
    t = read_nc_files(region, base_path, filename=files['t'], year_range=range(2000, 2002))
    v = read_nc_files(region, base_path, filename=files['v'], year_range=range(2000, 2002))
    p = read_nc_files(region, base_path, filename=files['pres'], year_range=range(2000, 2002))

    t = t.diff("time")
    v = v.diff("time")
    p = p.diff("time")
    print(t)
    print(v)
    print(p)

main()





