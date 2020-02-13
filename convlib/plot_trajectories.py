import xarray as xr
import matplotlib as mpl
from LCS import LCS, parcel_propagation

#mpl.use('Agg')
from convlib.xr_tools import read_nc_files, createDomains, add_basin_coord, get_xr_seq
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import sys

region = 'SACZ_big'
years = range(2020, 2021)
lcstimelen = 6

basin = 'Tiete'
season = 'DJF'
MAG = xr.open_dataset('data/xarray_mair_grid_basins.nc')
MAG = MAG[basin]
MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})
datapath_local = '/home/gab/phd/data/'

ftle_array = read_nc_files(region=region,
                           basepath=datapath_local + 'FTLE_ERA5/',
                           filename='SL_attracting_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                           year_range=years, season=None, lcstimelen=lcstimelen,
                           set_date=True)
ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)*6/lcstimelen
#ftle_array.roll(time=-2, roll_coords=False)
u = read_nc_files(region={'latitude': slice(10, -70),
                          'longitude': slice(-100, -1)},
                  basepath=datapath_local+'/ERA5/ERA5',
                  filename='viwve_ERA5_6hr_{year}010100-{year}123118.nc',
                  year_range=years, transformLon=True, reverseLat=False,
                  time_slice_for_each_year=slice(None, None), season=season)
v = read_nc_files(region={'latitude': slice(10, -70),
                          'longitude': slice(-100, -1)},
                  basepath=datapath_local+'/ERA5/ERA5',
                  filename='viwvn_ERA5_6hr_{year}010100-{year}123118.nc',
                  year_range=years, transformLon=True, reverseLat=False,
                  time_slice_for_each_year=slice(None, None), season=season)
tcwv = read_nc_files(region={'latitude': slice(10, -70),
                          'longitude': slice(-100, -1)},
                  basepath=datapath_local+'/ERA5/ERA5',
                  filename='tcwv_ERA5_6hr_{year}010100-{year}123118.nc',
                  year_range=years, transformLon=True, reverseLat=False,
                  time_slice_for_each_year=slice(None, None), season=season)
u = u/tcwv.values
v = v/tcwv.values
# u = u.sel(time='FEB2000')
# v = v.sel(time='FEB2000')
# ftle_array = ftle_array.sel(time='FEB2000')
# tcwv = tcwv.sel(time='FEB2000')



u = get_xr_seq(u, 'time', [x for x in range(lcstimelen)])
u = u.dropna(dim='time', how='any')
v = get_xr_seq(v, 'time', [x for x in range(lcstimelen)])
v = v.dropna(dim='time', how='any')

u.name = 'u'
v.name = 'v'
ds = xr.merge([u, v])
del u
del v
MAG = MAG.interp_like(ds, method='nearest')
#ds[basin] = (("latitude", "longitude"), MAG.values)
#ds = ds.where(ds[basin] == 1, drop=True)

print(ds)
ds_groups = list(ds.groupby('time'))
input_arrays = []
for label, group in ds_groups:  # have to do that because bloody groupby returns the labels
    input_arrays.append(group)
x_list = []
y_list = []
for i, input_array in enumerate(input_arrays):
    x_departure, y_departure = parcel_propagation(input_array.u.copy(), input_array.v.copy(), -6*3600, propdim='seq',
                                                  subtimes_len=1, return_traj=True)
    x_list.append(x_departure)
    y_list.append(y_departure)
    sys.stderr.write('\rdone {0:%}'.format(i / len(input_arrays)))
x_list = xr.concat(x_list, dim='time')
y_list = xr.concat(y_list, dim='time')
x_list.name = 'x_departure'
y_list.name = 'y_departure'
output = xr.merge([x_list, y_list])
dep_x = output.x_departure
dep_y = output.y_departure
dep_x = dep_x.assign_coords(seq=np.arange(lcstimelen+1), time=ds.time.copy())
dep_y = dep_y.assign_coords(seq=np.arange(lcstimelen+1), time=ds.time.copy())
MAG=MAG.sortby('latitude')

dep_x[basin] = (("latitude", "longitude"), MAG.values)
dep_y[basin] = (("latitude", "longitude"), MAG.values)
tcwv[basin] = (('latitude', 'longitude'), MAG.values)
tcwv=get_xr_seq(tcwv, 'time', [x for x in range(dep_x.seq.values.shape[0])])
fig=plt.figure(figsize=[20,20])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

for time in dep_x.time.values:
    dep_x_t = dep_x.sel(time=time).where(dep_x[basin]==1, drop=True)
    dep_y_t = dep_y.sel(time=time).where(dep_y[basin]==1, drop=True)
    tcwv_t = tcwv.sel(time=time)

    dpx = dep_x_t.longitude.values.flatten()
    dpy = dep_y_t.latitude.values.flatten()

    #fig = plt.figure(figsize=[20, 20])
    #ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    ax.coastlines(color='black', resolution='50m', linewidth=2)
    xx, yy = np.meshgrid(dpx[~np.isnan(dpx)], dpy[~np.isnan(dpy)])
    xx = xx.flatten()
    #tcwvalues = tcwv.values.flatten()[~np.isnan(tcwv.values.flatten())]
    yy = yy.flatten()
    xx = xx[range(0, xx.shape[0], 4)]
    yy = yy[range(0, yy.shape[0], 4)]

    positions = zip(xx.flatten(), yy.flatten())
    k = 0
    for posx, posy, in positions:
        print(posx, posy)
        ax.plot(dep_x_t.sel(latitude=posy, longitude=posx, method='nearest').values,
                dep_y_t.sel(latitude=posy, longitude=posx, method='nearest').values, c='black', linewidth=0.2)
        p=ax.scatter(dep_x_t.sel(latitude=posy, longitude=posx, method='nearest').values,
                   dep_y_t.sel(latitude=posy, longitude=posx, method='nearest').values,
                   c=tcwv_t.sel(latitude=posy, longitude=posx, method='nearest').differentiate('seq').values, cmap='RdBu', vmin=-5, vmax=5)
        # c=np.arange(len(dep_x.time.values)))
        k += 1

    plt.xlim([dep_x.longitude.values.min() + 10, dep_x.longitude.values.max() - 8])
    plt.ylim([dep_x.latitude.values.min() + 10, dep_x.latitude.values.max() - 4])
    ax.set_title(f'{time}')
    #fig.colorbar(p)

    plt.savefig(f'/home/users/gmpp/analysis_dump/figs/test_trajectories_{lcstimelen}_{time}_v2.png')
    #plt.close()


raise ValueError()


u = u.sel(time=slice(pd.Timestamp('2000-02-10T12:00:00') - pd.to_timedelta(str(lcstimelen*6-6)+"H"),
                     '2000-02-10T12:00:00'))
v = v.sel(time=slice(pd.Timestamp('2000-02-10T12:00:00') - pd.to_timedelta(str(lcstimelen*6-6)+"H"),
                     '2000-02-10T12:00:00'))
ftle_array = ftle_array.sel(time=slice(pd.Timestamp('2000-02-10T12:00:00') - pd.to_timedelta(str(lcstimelen*6 -6)+"H"),
                                       pd.Timestamp('2000-02-10T12:00:00')))
ftle = ftle_array.sel(time='2000-02-10T12:00:00')
dep_x, dep_y = parcel_propagation(u.copy(), v.copy(), timestep=-6*3600, return_traj=True, subtimes_len=3)
dep_x = dep_x.sel(latitude=ftle.latitude.values, longitude=ftle.longitude.values, method='nearest')
dep_y = dep_y.sel(latitude=ftle.latitude.values, longitude=ftle.longitude.values, method='nearest')
MAG = MAG.interp_like(dep_x, method='nearest')
dep_x[basin] = (("latitude", "longitude"), MAG.values)
dep_y[basin] = (("latitude", "longitude"), MAG.values)
ftle[basin] = (("latitude", "longitude"), MAG.values)

ftle = ftle.where(ftle[basin] == 1, drop=True)
dep_x = dep_x.where(ftle > 1.0, drop=True)
dep_y = dep_y.where(ftle > 1.0, drop=True)
tcwv = tcwv.sel(time=dep_x.time.values).where(ftle > 1.0, drop=True)

#dep_x = dep_x.sel(latitude=slice(-30, None), longitude=slice(-50, None))
#dep_y = dep_y.sel(latitude=slice(-30, None), longitude=slice(-50, None))

dpx = dep_x.longitude.values.flatten()
dpy = dep_y.latitude.values.flatten()

fig = plt.figure(figsize=[20, 20])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

ftle.plot.contourf(levels=np.arange(0, 2, 0.05), cmap="Greys_r", ax=ax, transform=ccrs.PlateCarree())
ftle.plot.contour(levels=[0.8999999, 0.9], ax=ax, colors=['red'], transform=ccrs.PlateCarree())

ax.coastlines(color='white', resolution='50m', linewidth=2)

xx, yy = np.meshgrid(dpx[~np.isnan(dpx)], dpy[~np.isnan(dpy)])
xx = xx.flatten()
tcwvalues = tcwv.values.flatten()[~np.isnan(tcwv.values.flatten())]
yy = yy.flatten()
xx = xx[range(0, xx.shape[0], 4)]
yy = yy[range(0, yy.shape[0], 4)]

positions = zip(xx.flatten(), yy.flatten())
k=0
for posx, posy, in positions:
    print(posx, posy)
    ax.plot(dep_x.sel(latitude=posy, longitude=posx, method='nearest').values,
             dep_y.sel(latitude=posy, longitude=posx, method='nearest').values, c='black')
    ax.scatter(dep_x.sel(latitude=posy, longitude=posx, method='nearest').values,
             dep_y.sel(latitude=posy, longitude=posx, method='nearest').values, s=tcwvalues[k],
               c=tcwv.sel(latitude=posy, longitude=posx, method='nearest').values)
              # c=np.arange(len(dep_x.time.values)))
    k+=1
plt.xlim([ftle.longitude.values.min()+10, ftle.longitude.values.max()-8])
plt.ylim([ftle.latitude.values.min()+10, ftle.latitude.values.max()-4])
plt.savefig(f'/home/users/gmpp/phdlib/convlib/tempfigs/diagnostics/test_trajectories_{lcstimelen}.png')
plt.close()

