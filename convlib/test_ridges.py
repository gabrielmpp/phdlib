from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import hessian
import cartopy.crs as ccrs

def gen_data(xx, yy):
    return np.cos(xx * np.pi/180)*np.cos(12*yy * np.pi/180)

dshape = [150, 100]
da = xr.DataArray(np.ones(dshape), dims=['latitude', 'longitude'], coords={
    'latitude': np.linspace(-50, 0, 150), 'longitude': np.linspace(-90, -30, 100)}
                  )

xx, yy = np.meshgrid(da.longitude.values, da.latitude.values)
da = da.copy(data=gen_data(xx, yy))
ridges, eigmin, eigvectors , gradient= find_ridges_spherical_hessian(da, sigma=None,
                                                                           scheme='second_order',
                                                                           return_eigvectors=True)
ridges = ridges.where(eigmin<eigmin.quantile(.1))
# gradient = gradient.copy(data=np.gradient(da.values))
# gradient = gradient.unstack()
np.abs(ridges).plot()
da.plot()
plt.show()

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
da.plot.contour(ax=ax)
da.where(ridges>0).plot(add_colorbar=True, ax=ax)

ax.quiver(eigvectors.longitude.values,
          eigvectors.latitude.values,
          eigvectors.isel(eigvectors=0).values * eigmin.values,
          eigvectors.isel(eigvectors=1).values * eigmin.values)
ax.quiver(gradient.longitude.values,
          gradient.latitude.values,
          gradient.isel(elements=0).values,
          gradient.isel(elements=1).values , color='red')

plt.savefig(f'test.png',
            dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)
plt.close()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, da.values.T, cmap='RdBu')
# ax.quiver(xx, yy, da.values.T+.1,  gradient.isel(elements=0).values.T, gradient.isel(elements=1).values.T,
#           gradient.isel(elements=0).values.T*0, length=10000, color='black')
# plt.savefig(f'test2.png',
#             dpi=1000, transparent=True, bbox_inches='tight', pad_inches=0)
# plt.close()