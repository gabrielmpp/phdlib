import xarray as xr
import matplotlib.pyplot as plt

filepath = '/home/users/gmpp/SL.nc'

array = xr.open_dataarray(filepath)

for time in array.time.values:
    plt.figure(figsize=[13,13])
    array.sel(time=time).plot(vmin=0,vmax=0.5,cmap = 'nipy_spectral')
    plt.savefig(f'./tempfigs/SL{time}.png')
    plt.close()