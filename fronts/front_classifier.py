import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
base_path = '/gws/nopw/j04/primavera1/observations/ERA5/'
files = dict(pres='sp_ERA5_6hr_{year}010100-{year}123118.nc',
             v='v_925_ERA5_6hrPlevPt_{year}010100-{year}123118.nc',
             t='t_700_ERA5_6hrPlevPt_{year}010100-{year}123118.nc')

domains = dict(
    AITCZ=dict(latitude=slice(-5, 15), longitude=slice(-50, -13)),
    SACZ=dict(latitude=slice(-5,-45), longitude=slice(-80,-10)))

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
        array.coords['longitude'].values = (array.coords['longitude'].values + 180) % 360 - 180
        array = array.sel(domains[region])
        array = array.resample(time="1D").mean('time')
        file_list.append(array)
    full_array = xr.concat(file_list, dim='time')
    print('*---- Finished reading data ----*')
    return full_array

def main():
    region = "SACZ"
    t = read_nc_files(region, base_path, filename=files['t'], year_range=range(1980, 2008))
    v = read_nc_files(region, base_path, filename=files['v'], year_range=range(1980, 2008))
    p = read_nc_files(region, base_path, filename=files['pres'], year_range=range(1980, 2008))

    t = t.diff("time")
    v = xr.apply_ufunc(lambda x: np.sign(x), v)
    v = v.diff("time")
    p = p.diff("time")
    p = p.where(p > 0, 0)
    p = p.where(np.abs(v)==2, 0)
    p = p.where(t < 0, 0)
    final = p.where(p == 0, 1)

    # --- Plot --- #
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    final.sum('time').plot(ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.savefig("temp.png")


    print(t)
    print(v)
    print(p)

main()





