import xarray as xr
import matplotlib.pyplot as plt

filepath = '/home/users/gmpp/SL.nc'

array = xr.open_dataarray(filepath)

for time in array.time.values:
    plt.figure(figsize=[13,13])
    array = array ** 0.5
    array.sel(time=time).plot.contourf(vmin=0,vmax=1,cmap = 'nipy_spectral')
    plt.savefig(f'./tempfigs/SL{time}.png')
    plt.close()