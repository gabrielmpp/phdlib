import cartopy.crs as ccrs
import xarray as xr
from LagrangianCoherence.LCS import LCS
import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr

from xr_tools.tools import filter_ridges
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
from cartopy.io.img_tiles import Stamen
from matplotlib.gridspec import GridSpec as GS
import sys
import pandas as pd
import matplotlib


def find_ridges_spherical_hessian(da, sigma=.5, scheme='first_order',
                                  angle=5):
    """
    Method to in apply a Hessian filter in spherical coordinates
    Parameters
    ----------
    sigma - float, smoothing intensity
    da - xarray.dataarray
    scheme - str, 'first_order' for x[i+1] - x[i] and second order for x[i+1] - x[i-1]
    angle - float in degrees to relax height perpendicularity requirement
    Returns
    -------
    Filtered array

    """
    da = da.copy()
    # Gaussian filter
    if isinstance(sigma, (float, int)):
        da = da.copy(data=gaussian_filter(da, sigma=sigma))

    # Initializing
    earth_r = 6371000
    x = da.longitude.copy() * np.pi / 180
    y = da.latitude.copy() * np.pi / 180
    dx = x.diff('longitude') * earth_r * np.cos(y)
    dy = y.diff('latitude') * earth_r
    dx_scaling = 2 * da.longitude.diff('longitude').values[0]  # grid spacing for xr.differentiate
    dy_scaling = 2 * da.latitude.diff('latitude').values[0]  # grid spacing
    print(1)
    # Calc derivatives
    if scheme == 'second_order':
        ddadx = dx_scaling * da.differentiate('longitude') / dx
        ddady = dy_scaling * da.differentiate('latitude') / dy
        d2dadx2 = dx_scaling * ddadx.differentiate('longitude') / dx
        d2dadxdy = dy_scaling * ddadx.differentiate('latitude') / dy
        d2dady2 = dx_scaling * ddady.differentiate('latitude') / dy
        d2dadydx = d2dadxdy.copy()
    elif scheme == 'first_order':
        ddadx = da.diff('longitude') / dx
        ddady = da.diff('latitude') / dy
        d2dadx2 = ddadx.diff('longitude') / dx
        d2dadxdy = ddadx.diff('latitude') / dy
        d2dady2 = ddady.diff('latitude') / dy
        d2dadydx = d2dadxdy.copy()
    # Assembling Hessian array
    print(2)
    gradient = xr.concat([ddadx, ddady],
                         dim=pd.Index(['ddadx', 'ddady'],
                                      name='elements'))
    hessian = xr.concat([d2dadx2, d2dadxdy, d2dadydx, d2dady2],
                        dim=pd.Index(['d2dadx2', 'd2dadxdy', 'd2dadydx', 'd2dady2'],
                                     name='elements'))
    hessian = hessian.stack({'points': ['latitude', 'longitude']})
    gradient = gradient.stack({'points': ['latitude', 'longitude']})
    hessian = hessian.where(np.abs(hessian) != np.inf, np.nan)
    hessian = hessian.dropna('points', how='any')
    gradient = gradient.sel(points=hessian.points)
    grad_vals = gradient.transpose(..., 'points').values
    hess_vals = hessian.transpose(..., 'points').values

    hess_vals = hess_vals.reshape([2, 2, hessian.shape[-1]])
    val_list = []
    eigmin_list = []
    print('Computing hessian eigvectors')
    for i in range(hess_vals.shape[-1]):
        print(str(100 * i / hess_vals.shape[-1]) + ' %')
        eig = np.linalg.eig(hess_vals[:, :, i])
        eigvector = eig[1][np.argmax(eig[0])]  # eigenvetor of smallest eigenvalue
        # eigvector = eigvector/np.max(np.abs(eigvector))  # normalizing the eigenvector to recover t hat

        dt_angle = np.arccos(np.dot(np.flip(eigvector), grad_vals[:, i]) / (np.linalg.norm(eigvector) *
                                                                            np.linalg.norm(grad_vals[:, i])))
        val_list.append(dt_angle)
        eigmin_list.append(np.min(eig[0]))

    dt_prod = hessian.isel(elements=0).drop('elements').copy(data=val_list).unstack()
    dt_prod_ = dt_prod.copy()
    eigmin = hessian.isel(elements=0).drop('elements').copy(data=eigmin_list).unstack()

    dt_prod = dt_prod.where(np.abs(dt_prod_) <= angle * np.pi / 180, 0)
    dt_prod = dt_prod.where(np.abs(dt_prod_) > angle * np.pi / 180, 1)
    dt_prod = dt_prod.where(eigmin < 0, 0)

    return dt_prod, eigmin

# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_6hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_6hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_6hr_2020010100-2020123118.nc'
evap_filepath = basepath + 'evap_ERA5_6hr_2020010100-2020123118.nc'
pr_filepath = basepath + 'pr_ERA5_6hr_2020010100-2020123118.nc'

# timesel = sys.argv[1]
timesel = slice('2020-01-20', '2020-01-28')
subdomain = {'latitude': slice(-40, 15),
             'longitude': slice(-85, -32)}
u = xr.open_dataarray(u_filepath, chunks={'time': 140})
u = u.assign_coords(longitude=(u.coords['longitude'].values + 180) % 360 - 180)
u = u.sortby('longitude')
u = u.sortby('latitude')

u = u.sel(latitude=slice(-85, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
u = u.sel(time=timesel)
u = u.load()

v = xr.open_dataarray(v_filepath, chunks={'time': 140})
v = v.assign_coords(longitude=(v.coords['longitude'].values + 180) % 360 - 180)
v = v.sortby('longitude')
v = v.sortby('latitude')
v = v.sel(latitude=slice(-85, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
v = v.sel(time=timesel)
v = v.load()

tcwv = xr.open_dataarray(tcwv_filepath, chunks={'time': 140})
tcwv = tcwv.assign_coords(longitude=(tcwv.coords['longitude'].values + 180) % 360 - 180)
tcwv = tcwv.sortby('longitude')
tcwv = tcwv.sortby('latitude')
tcwv = tcwv.sel(latitude=slice(-88, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
tcwv = tcwv.sel(time=timesel)
tcwv = tcwv.load()

evap = xr.open_dataarray(evap_filepath, chunks={'time': 140})
evap = evap.assign_coords(longitude=(evap.coords['longitude'].values + 180) % 360 - 180)
evap = evap.sortby('longitude')
evap = evap.sortby('latitude')
evap = evap.sel(latitude=slice(-88, 35), longitude=slice(-170, 30)).sel(expver=1).drop('expver')
evap = evap.sel(time=timesel)
evap = evap * 1000
evap = evap.load()

pr = xr.open_dataarray(pr_filepath, chunks={'time': 140})
pr = pr.assign_coords(longitude=(pr.coords['longitude'].values + 180) % 360 - 180)
pr = pr.sortby('longitude')
pr = pr.sortby('latitude')
pr = pr.sel(subdomain)
pr = pr.sel(time=timesel)
pr = pr.load()
pr = pr * 1000  # Originally in metres
pr = pr #  Converting to mm/h
pr.isel(time=19).plot.contour(levels=[1, 5, 10], cmap=cmr.freeze)
plt.show()
# rain = -u.sel(subdomain).differentiate('longitude') - v.sel(subdomain).differentiate('latitude')

u = u/tcwv
v = v/tcwv
u.name = 'u'
v.name = 'v'
ds = xr.merge([u, v])
# ---- Running LCS ---- #
ntimes = 30*4
ftle_list = []



for dt in range(0, ntimes, 4):
    dt=12
    timeseq = np.arange(0, 8) + dt
    # lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, return_det=True)
    # det = lcs(ds.isel(time=timeseq))
    # det = np.sqrt(det)
    # potential_rainfall = (tcwv.isel(time=3) / (-6*3600*timeseq.shape[0])) * (1 - det)/det
    # potential_rainfall = - ( 2 * tcwv / (-6*3600*5) ) * (det - 1) / (det + 1)  # [mm / s]
    # potential_rainfall = potential_rainfall * 86400  # conversion to [mm / day]

    lcs = LCS.LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, return_dpts=True,
                  gauss_sigma=None)
    ftle, x_departure, y_departure = lcs(ds.isel(time=timeseq), s=1e5, resample='3H')

    x_departure = x_departure.sel(subdomain)
    y_departure = y_departure.sel(subdomain)
    from scipy.ndimage import gaussian_filter

    ftle_ = ftle.isel(time=0)

    ftle_ = (4 / timeseq.shape[0]) * np.log(ftle_)

    ftle_ = ftle_.sortby('longitude')
    ftle_ = ftle_.sortby('latitude')
    ftle_.plot.contour(levels=[1],vmin=1, cmap='Reds')
    pr.isel(time=timeseq[-1]).plot(vmax=10)
    plt.show()
    tcwv_ = tcwv.isel(time=timeseq).sel(latitude=ftle_.latitude.values, longitude=ftle_.longitude.values, method='nearest')
    dx =  tcwv_.longitude.diff('longitude') * 6370000 * np.pi/180
    dy =  tcwv_.latitude.diff('latitude') * 6370000 * np.pi/180
    tcwv_dx = tcwv_.diff('longitude') / dx
    tcwv_dy = tcwv_.diff('latitude')  / dy
    tcwv_dx, tcwv_dy = xr.align(tcwv_dx, tcwv_dy)
    gradient = np.sqrt(tcwv_dx ** 2 + tcwv_dy ** 2)
    tcwv_dx = tcwv_dx * ds['u'] * 3600
    tcwv_dy = tcwv_dy * ds['v'] * 3600
    tcwv_dx = tcwv_dx.isel(time=-1)
    tcwv_dy = tcwv_dy.isel(time=-1)
    adv = - (tcwv_dx + tcwv_dy)
    tcwv_dx = tcwv_dx.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    tcwv_dy = tcwv_dy.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    div_mass = tcwv_ * (ds['u'].diff('longitude')/dx + ds['v'].diff('latitude')/dy) * 3600

    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': ccrs.PlateCarree()})
    p = adv.plot(robust=True, ax=axs[0, 0], vmin=-2, vmax=2, add_colorbar=False,
                 cmap='RdBu')
    axs[0, 0].quiver(tcwv_dx.longitude.values, tcwv_dy.latitude.values,
               -tcwv_dx.values, -tcwv_dy.values)
    axs[0, 0].coastlines()
    axs[0, 0].set_title('Advection')
    (evap*-1).isel(time=timeseq[-1]).sel(latitude=ftle_.latitude.values,
                                    longitude=ftle_.longitude.values,
                                    method='nearest').plot(ax=axs[0, 1],
                                                           vmin=-2, vmax=2,
                                                           add_colorbar=False,
                                                           cmap='RdBu')
    axs[0, 1].coastlines()
    axs[0, 1].set_title('Evaporation')

    (-1*div_mass).isel(time=-1).plot(ax=axs[1, 0], vmin=-2, vmax=2, add_colorbar=False,
                                cmap='RdBu')
    axs[1, 0].coastlines()
    axs[1, 0].set_title('Mass Convergence')
    (-1*pr).isel(time=timeseq[-1]).plot(ax=axs[1, 1], robust=True, cmap='RdBu',
                                   vmin=-2, vmax=2, add_colorbar=False)
    axs[1, 1].set_title('Rain')
    axs[1, 1].coastlines()
    plt.colorbar(p, ax=axs[:, 1])
    plt.savefig(f'convlib/tempfigs/budget/{dt}.png', dpi=600,
            transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close()


    # residual = -evap.isel(time=timeseq).mean('time') - \
    #            pr.isel(time=timeseq).mean('time') - \
    #            div_mass.mean('time')
    # dqdt = tcwv_.isel(time=-1) - tcwv_.isel(time=0)
    # dqdt.plot(robust=True, vmin=-16, vmax=16, cmap=cmr.redshift)
    # plt.show()
    # residual.plot(robust=True, vmin=-6, vmax=6, cmap=cmr.redshift)
    # plt.show()


    # gradient = (gradient.isel(time=-1)-gradient.isel(time=0))
    # ftle__ = ftle_.interp(latitude=gradient.latitude,
    #                                       longitude=gradient.longitude,
    #                                       method='linear')
    # gradient, ftle__ = xr.align(gradient, ftle__)
    # ypts=gradient.stack({'samples':['latitude', 'longitude']}).values
    # xpts=ftle__.stack({'samples':['latitude', 'longitude']}).values
    # plt.scatter(xpts, ypts)
    # plt.ylim([-0.001, 0.001])
    # plt.hlines(y=0, xmin=-3, xmax=3)
    # plt.show()



    delta_tcwv = tcwv.isel(time=timeseq[-1]) - tcwv.isel(time=timeseq[0])
    delta_tcwv = delta_tcwv.sel(latitude=ftle_.latitude, longitude=ftle_.longitude, method='nearest')
    delta_tcwv = delta_tcwv + pr.isel(time=timeseq).sum('time') \
    + evap.isel(time=timeseq).sum('time')

    fig = plt.figure(figsize=[20, 5])
    axs = []

    gs = matplotlib.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])
    axs.append(fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(gs[0, 1]))
    p = delta_tcwv.plot(robust=True, ax=axs[0], add_colorbar=False,
                       transform=ccrs.PlateCarree())
    ftle_.plot.contour(levels=[.8, 1, 2, 3], cmap=cmr.rainforest, ax=axs[0],
                       transform=ccrs.PlateCarree())
    axs[0].coastlines()
    axs[0].set_title('')
    delta_tcwv, ftle_ = xr.align(delta_tcwv, ftle_)
    ypts=delta_tcwv.stack({'samples':['latitude', 'longitude']}).values
    xpts=ftle_.stack({'samples':['latitude', 'longitude']}).values

    axs[1].scatter(xpts, ypts, alpha=0.5)
    axs[1].set_title('')
    axs[1].set_ylabel(r'$\Delta TCWV$')
    axs[1].set_xlabel(r'FTLE ($\frac{1}{day}$)')
    # axs[1].set_xlim([-2, 4])
    cbar=plt.colorbar(p, ax=axs[0], orientation='vertical', shrink=0.6)
    cbar.ax.set_ylabel(r'$\Delta TCWV$')
    plt.savefig(f'convlib/tempfigs/budget/delta_{dt}.png', dpi=600,
                transparent=False, bbox_inches='tight')#, pad_inches=0.1)
    plt.close()

    delta_tcwv = tcwv.isel(time=timeseq[-1]) - tcwv.isel(time=timeseq[-2])
    delta_tcwv = delta_tcwv.sel(latitude=ftle_.latitude, longitude=ftle_.longitude, method='nearest')
    delta_tcwv = delta_tcwv + pr.isel(time=timeseq).sum('time') \
    + evap.isel(time=timeseq).sum('time')
    fig = plt.figure(figsize=[20, 5])
    axs = []

    gs = matplotlib.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.5, 1])
    axs.append(fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(gs[0, 1]))
    p = delta_tcwv.plot(robust=True, ax=axs[0], add_colorbar=False,
                       transform=ccrs.PlateCarree())
    ftle_.plot.contour(levels=[.8, 1, 2, 3], cmap=cmr.rainforest, ax=axs[0],
                       transform=ccrs.PlateCarree())
    axs[0].coastlines()
    axs[0].set_title('')
    delta_tcwv, ftle_ = xr.align(delta_tcwv, ftle_)
    ypts=delta_tcwv.stack({'samples':['latitude', 'longitude']}).values
    xpts=ftle_.stack({'samples':['latitude', 'longitude']}).values

    axs[1].scatter(xpts, ypts, alpha=0.5)
    axs[1].set_title('')
    axs[1].set_ylabel(r'$\Delta TCWV$')
    axs[1].set_xlabel(r'FTLE ($\frac{1}{day}$)')
    # axs[1].set_xlim([-2, 4])
    cbar=plt.colorbar(p, ax=axs[0], orientation='vertical', shrink=0.6)
    cbar.ax.set_ylabel(r'$\Delta TCWV$')
    plt.savefig(f'convlib/tempfigs/budget/delta_inst{dt}.png', dpi=600,
                transparent=False, bbox_inches='tight')#, pad_inches=0.1)
    plt.close()