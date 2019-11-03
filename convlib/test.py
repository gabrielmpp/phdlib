
import xarray as xr
from convlib.LCS import parcel_propagation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lat1 = -20
lat2 = -5
lon1 = -60
lon2 = -30
dx = 0.25
earth_r = 6371000

nlat = int((lat2-lat1)/dx)
nlon = int((lon2-lon1)/dx)
latitude = np.linspace(lat1, lat2, nlat)
longitude = np.linspace(lon1,lon2, nlon)
time = pd.date_range("2000-01-01", "2000-01-02", freq="6H")
u = xr.DataArray(10, dims=['latitude', 'longitude', 'time'],
                  coords={'latitude': latitude, 'longitude': longitude, 'time': time})
v = xr.DataArray(0, dims=['latitude', 'longitude', 'time'],
                  coords={'latitude': latitude, 'longitude': longitude, 'time': time})

dep_x, dep_y = parcel_propagation(u, v, timestep=6*3600, subtimes_len=1)
origin = np.meshgrid(longitude, latitude)[0]
origin.shape
displacement = dep_x.copy(data=dep_x - origin)
plt.streamplot(longitude, latitude, u.isel(time=0).values, v.isel(time=1).values)
(displacement/len(time)).plot()
plt.show()
dep_y.plot()
plt.show()
#
conversion_y = 1 / earth_r  # converting m/s to rad/s
conversion_x = (earth_r ** (-1)) * xr.apply_ufunc(lambda x: np.cos(x * np.pi / 180), u.latitude) ** (-1)
conversion_x, _ = xr.broadcast(conversion_x, u.isel({'time': 0}))
conversion_x.plot()
plt.plot(conversion_x)
plt.show()

ftle = xr.open_dataarray('/run/media/gab/gab_hd/scripts/phdlib/convlib/data/SL_repelling_2000_lcstimelen_1_v2.nc')

for time in ftle.time.values:
    plt.figure(figsize=[10,10])
    ftle.sel(time=time).plot(cmap='nipy_spectral', vmax=6,vmin=0)
    plt.show()