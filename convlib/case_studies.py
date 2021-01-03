from LagrangianCoherence.LCS.area_of_influence import find_area
import xarray as xr
import numpy as np
from LagrangianCoherence.LCS.LCS import LCS
from LagrangianCoherence.LCS.tools import find_ridges_spherical_hessian
from xr_tools.tools import filter_ridges
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib
import time
import pandas as pd
import cmasher as cmr
from skimage.morphology import skeletonize, binary_erosion, binary_dilation, medial_axis
from skimage.filters import threshold_local
from scipy import ndimage as ndi
from LagrangianCoherence.LCS.LCS import compute_deftensor
from matplotlib.gridspec import GridSpec as GS
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
from numpy.linalg import norm
import sys
from skimage.filters import gaussian


def to_potential_temp(tmp, pres):
    return tmp * (100000 / pres) ** 0.286


def calc_def_tensor(da_1, da_2, sigma=1.5):
    # sigma=5 for temperature
    def_tensor = compute_deftensor(da_1, da_2, sigma=sigma)
    def_tensor = def_tensor.stack({'points': ['latitude', 'longitude']})
    def_tensor = def_tensor.dropna(dim='points')

    vals = def_tensor.transpose(..., 'points').values
    vals = vals.reshape([2, 2, def_tensor.shape[-1]])
    def_tensor_norm_tmp = norm(vals, axis=(0, 1), ord=2)
    def_tensor_norm_tmp = def_tensor.isel(derivatives=0).drop('derivatives').copy(data=def_tensor_norm_tmp)
    def_tensor_norm_tmp = def_tensor_norm_tmp.unstack('points')
    return def_tensor_norm_tmp

event_subtropical_cyclone = slice('2020-06-25', '2020-07-05')
event_ITCZ = slice('2020-04-10', '2020-04-18')
event_ITCZ2 = slice('2020-03-12', '2020-03-18')
event_jbn = slice('2020-05-07', '2020-05-18')
event_front_winter = slice('2020-06-22', '2020-07-02')
event_front_summer = slice('2020-01-12', '2020-01-20')
event_sacz = slice('2020-01-18T18:00:00', '2020-01-26')
# casestudy = str(sys.argv[1])
casestudy = 'sacz'
if casestudy == 'itcz':
    event = event_ITCZ
elif casestudy == 'itcz2':
    event = event_ITCZ2
elif casestudy =='subtropical':
    event = event_subtropical_cyclone
elif casestudy == 'jbn':
    event = event_jbn
elif casestudy == 'front_summer':
    event = event_front_summer
elif casestudy == 'front_winter':
    event = event_front_winter
elif casestudy == 'sacz':
    event = event_sacz


experiment_2days_path = '/group_workspaces/jasmin4/upscale/gmpp/convzones/' \
                  'experiment_timelen_8_db52bba2-b71a-4ab6-ae7c-09a7396211d4'

case_study_outpath = '/home/gab/phd/data/case_studies/'

u_filename = 'ERA5viwve_ERA5_6hr_{year}010100-{year}123118.nc'
v_filename = 'ERA5viwvn_ERA5_6hr_{year}010100-{year}123118.nc'
tcwv_filename = 'ERA5tcwv_ERA5_6hr_{year}010100-{year}123118.nc'

# ---- Preparing input ---- #
basepath = '/home/gab/phd/data/ERA5/'
u_filepath = basepath + 'viwve_ERA5_3hr_2020010100-2020123118.nc'
v_filepath = basepath + 'viwvn_ERA5_3hr_2020010100-2020123118.nc'
tcwv_filepath = basepath + 'tcwv_ERA5_3hr_2020010100-2020123118.nc'
tmp_filepath = basepath + 'tmp_2m_ERA5_6hr_2020010100-2020123118.nc'
mslpres_filepath = basepath + 'mslpres_ERA5_6hr_2020010100-2020123118.nc'
sfcpres_filepath = basepath + 'sfcpres_ERA5_6hr_2020010100-2020123118.nc'
orog_filepath = basepath + 'geopotential_orography.nc'
data_dir = '/home/gab/phd/data/composites_cz/'
pr_filepath = basepath + 'pr_ERA5_6hr_2020010100-2020123118.nc'

subdomain = {'latitude': slice(-40, 15),
             'longitude': slice(-90, -32)}

def read_file(filepath):
    try:
        da = xr.open_dataarray(filepath, chunks={'time': 50})
    except ValueError:
        da = xr.open_dataset(filepath, chunks={'time': 50})['t2m_0001']
    da = da.assign_coords(longitude=(da.coords['longitude'].values + 180) % 360 - 180)
    da = da.sortby('longitude')
    da = da.sortby('latitude')
    return da


u = read_file(u_filepath).sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
v = read_file(v_filepath).sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
tcwv = read_file(tcwv_filepath).sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
mslpres = read_file(mslpres_filepath).sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
sfcpres = read_file(sfcpres_filepath).sel(latitude=slice(-75, 60), longitude=slice(-160, 45)).sel(expver=1).drop('expver')
tmp = read_file(tmp_filepath).sel(latitude=slice(-75, 60), longitude=slice(-150, 45))
pr = read_file(pr_filepath).sel(latitude=slice(-75, 60), longitude=slice(-150, 45)).sel(expver=1).drop('expver')
pr = pr * 3600
orog = read_file(orog_filepath).sel(latitude=slice(-56, 15), longitude=slice(-90, -30))
orog = orog / 9.80665  # grav
sealandmask = orog.where(orog>10, 0).isel(time=0)
sealandmask = sealandmask.where(sealandmask < 10, 1)

u_unscaled = u.copy()
v_unscaled = v.copy()
u = u / tcwv
v = v / tcwv
u.name = 'u'
v.name = 'v'



timesel = event

u = u.sel(time=timesel)
v = v.sel(time=timesel)
u_unscaled = u_unscaled.sel(time=timesel)
v_unscaled = v_unscaled.sel(time=timesel)
tmp = tmp.sel(time=timesel)
tcwv = tcwv.sel(time=timesel)
mslpres = mslpres.sel(time=timesel)
sfcpres = sfcpres.sel(time=timesel)
pr = pr.sel(time=timesel)
pr = pr.load()
u = u.load()
u_unscaled = u_unscaled.load()
v_unscaled = v_unscaled.load()
mslpres = mslpres.load()
sfcpres = sfcpres.load()
tcwv = tcwv.load()
v = v.load()
tmp = tmp.load()
dt=17
flux_mag = (u_unscaled**2)**0.5 + (v_unscaled**2) ** 0.5

for dt in range(1, u.time.values.shape[0], 1):
    time1 = time.time()
    font = {'size': 10}

    matplotlib.rc('font', **font)
    timeseq = np.arange(0, 16) + dt
    ds = xr.merge([u, v])
    ds = ds.isel(time=timeseq)
    lcs = LCS(timestep=-3 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
              gauss_sigma=None, return_dpts=True)
    #  Good error s=0.00262
    lcs_local = LCS(timestep=-3 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
                    gauss_sigma=None, return_dpts=False)
    lcs_12hrs = LCS(timestep=-3 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain,
                    gauss_sigma=None, return_dpts=False)

    ftle, x_departure, y_departure = lcs(ds, s=.25e6, s_is_error=False) #resample='3H')
    ftle_local = lcs_local(ds.isel(time=[-2, -1]), s=1e6 / 16, s_is_error=False, resample=None)
    ftle_12hrs = lcs_12hrs(ds.isel(time=[-4, -3, -2, -1]), s=1e6 / 8, s_is_error=False, resample=None)
    ftle = np.log(ftle) / 2
    ftle_local = np.log(ftle_local) * 4
    ftle_12hrs = np.log(ftle_12hrs) * 2
    ftle_local = ftle_local.isel(time=0)
    ftle = ftle.isel(time=0)
    ftle_12hrs = ftle_12hrs.isel(time=0)

    local_thresh = ftle_local.copy(data=threshold_local(ftle_local.values, 301,
                                                        offset=-.8))
    binary_local = ftle_local > local_thresh
    ftle_local_high = binary_local
    distance = binary_local
    ridges, eigmin, eigvectors, gradient = find_ridges_spherical_hessian(ftle,
                                                               sigma=5,
                                                               tolerance_threshold=0.002e-3,
                                                               scheme='second_order',
                                                               return_eigvectors=True)
    ridges_12, eigmin_12, eigvectors_12, gradient_12 = find_ridges_spherical_hessian(ftle_12hrs,
                                                               sigma=5,
                                                               tolerance_threshold=0.001e-3,
                                                               scheme='second_order',
                                                               return_eigvectors=True)
    ridges = ridges.where(ftle > 1.2, 0)
    ridges_12 = ridges_12.where(ftle_12hrs > 2.2, 0)
    threshold = -5.23023852e-11 # quantile 0.1
    ridges = ridges.where(eigmin<threshold, 0)
    ridges_12 = ridges_12.where(eigmin_12<threshold, 0)
    ridges = filter_ridges(ridges, ftle, criteria=['major_axis_length'],
                           thresholds=[30])
    ridges_12 = filter_ridges(ridges_12, ftle_12hrs, criteria=['major_axis_length'],
                           thresholds=[15])
    # ridges = ridges.copy(data=skeletonize(ridges.values))
    # ridges.plot()
    # plt.show()
    # ridges = ridges.copy(data=binary_dilation(ridges.values))
    # ridges.plot()
    # plt.show()
    ridges = ridges.where(~xr.ufuncs.isnan(ridges), 0)
    ridges_12 = ridges_12.where(~xr.ufuncs.isnan(ridges), 0)

    mslpres_ = mslpres.interp(time=u.time, method='linear').isel(time=timeseq[-1]).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    sfcpres_ = sfcpres.interp(time=u.time, method='linear').isel(time=timeseq[-1]).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    tcwv_ = tcwv.isel(time=timeseq[-1]).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    tmp_ = tmp.interp(time=u.time, method='linear').isel(time=timeseq[-1]).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    u_unscaled_ = u_unscaled.isel(time=timeseq[-1]).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    v_unscaled_ = v_unscaled.isel(time=timeseq[-1]).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    tmp_ = to_potential_temp(tmp_, sfcpres_)  - 273.15
    # tmp_ = tmp_.copy(data=gaussian(tmp_.values, sigma=2))

    dpdx = mslpres_.differentiate('longitude')
    dpdy = mslpres_.differentiate('latitude')
    u_vec = eigvectors.isel(eigvectors=0)
    v_vec = eigvectors.isel(eigvectors=1)
    pres_gradient_parallel = np.sqrt((dpdx * v_vec) ** 2 + (dpdy * u_vec) ** 2)

    ridges_pres_grad = ridges * pres_gradient_parallel

    ridges_pres_grad = filter_ridges(ridges, ridges_pres_grad, criteria=['mean_intensity'],
                                     thresholds=[0.1])

    def_tensor_norm_tmp = calc_def_tensor(tmp_, tmp_, sigma=2)

    ridges_tmp, eivals_temp, eigvectors_tmp, gradient_tmp_max = find_ridges_spherical_hessian(def_tensor_norm_tmp,
                                                               sigma=1.2,
                                                               scheme='second_order',
                                                               return_eigvectors=True)

    u_tmp = eigvectors_tmp.isel(eigvectors=0)
    v_tmp = eigvectors_tmp.isel(eigvectors=1)
    u_tmp_lowres = u_tmp.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    v_tmp_lowres = v_tmp.coarsen(latitude=5, longitude=5, boundary='trim').mean()

    test = (((u_tmp * u_vec) + (v_tmp * v_vec))**2)**.5 * def_tensor_norm_tmp
    flux_along_ridge = (((u_unscaled_ * u_vec) + (v_unscaled_ * v_vec))**2)**.5


    ridges_bool = ridges == 1
    dist = ftle_local_high.copy(data=ndi.distance_transform_edt(~ridges_bool))

    ridges_dil = ridges.copy(data=binary_dilation(ridges.values))
    ridges_dil = ridges_dil.where(ridges_dil > 0)
    ridges = ridges.where(ridges > 0)
    ridges_pres_grad = ridges_pres_grad.where(ridges_pres_grad > 0)
    ridges_pres_grad = ridges_pres_grad.where(xr.ufuncs.isnan(ridges_pres_grad), 1)

    ridges_ = filter_ridges(ftle_local_high, ridges_dil.where(~xr.ufuncs.isnan(ridges_dil), 0),
                            criteria=['max_intensity'],
                            thresholds=[.5])

    ridges_coastline= filter_ridges(ridges.sel(latitude=test.latitude, longitude=test.longitude, method='nearest'), test,
                            criteria=['mean_intensity'],
                            thresholds=[.01])

    ridges_ = ridges_ * dist.where(dist < 12)
    ridges_ = ridges_.where(xr.ufuncs.isnan(ridges_), 1)
    local_strain = ftle_local_high - ridges_.where(~xr.ufuncs.isnan(ridges_), 0)

    pr_ = pr.interp(time=u.time, method='linear').sel(time=ridges.time).interp(method='nearest',
                                              latitude=ridges.latitude,
                                              longitude=ridges.longitude)
    flux_mag_ = flux_mag.sel(time=ridges.time).interp(method='nearest',
                                              latitude=ridges.latitude,
                                              longitude=ridges.longitude)
    uwind= (ds.u).isel(time=-1).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear') #* tcwv_
    uwind=  uwind.coarsen(latitude=5, longitude=5, boundary='trim').mean()

    vwind=ds.v.isel(time=-1).interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')#* tcwv_
    vwind= vwind.coarsen(latitude=5, longitude=5, boundary='trim').mean()

    #### ------  Plotting

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    # tmp_.plot(cmap=cmr.pride, ax=ax, vmax=35, vmin=10,
    #                                                linewidth=.5,
    #                                                add_colorbar=False)
    pr_.name = 'Precipitation (mm/h)'
    pr_.plot(add_colorbar=True, vmax=45, cmap=cmr.freeze_r, ax=ax, vmin=0)
    matplotlib.rcParams['hatch.color'] = 'blue'
    matplotlib.rcParams['hatch.linewidth'] = .3

    local_strain.plot.contourf(cmap='Reds', alpha=0,
                                   levels=[0, .5],
                                   add_colorbar=False,
                                   hatches=['',
                                            '////////'],
                                   ax=ax)
    matplotlib.rcParams['hatch.color'] = 'red'
    ridges_.where(~xr.ufuncs.isnan(ridges_), 0).plot.contourf(cmap='Reds', alpha=0,
                                                                  levels=[0, .5],
                                                                  add_colorbar=False,
                                                                  hatches=['',
                                                                           'xxxxxxx'],
                                                                  ax=ax)
    ridges.plot(add_colorbar=False, cmap=cmr.sunburst)
    ridges_12.plot(add_colorbar=False, cmap=cmr.freeze)
    # test.where(ridges>0).plot(add_colorbar=False, cmap=cmr.gem, ax=ax, vmax=10, vmin=0)
    # flux_along_ridge.where(ridges>0).plot(add_colorbar=False, cmap=cmr.gem, ax=ax, robust=True)#, vmax=1600, vmin=600)
    ax.coastlines(color='black')
    total_rain = np.sum(pr_)
    czs_rain = np.sum(ridges_ * pr_)
    local_strain_rain = np.sum(local_strain * pr_)
    rest = total_rain - local_strain_rain - czs_rain
    ax.text(-90, -39, str(np.round(czs_rain.values / 1000)) + ' m rain on red areas\n' +
                str(np.round(local_strain_rain.values / 1000)) + ' m rain on blue areas\n' +
                str(np.round(rest.values / 1000)) + 'm remaining',
                bbox={'color': 'black', 'alpha': .3},
                color='black',
                fontsize=6)
    plt.savefig(f'case_studies/{casestudy}/fig_temp_{dt:02d}_area.png', dpi=600,
                    transparent=True, pad_inches=0, bbox_inches='tight'
                    )
    plt.close()

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})


    p = flux_mag_.plot.contourf(levels=11, cmap=cmr.gem, ax=ax,vmax=1200, vmin=200,
                                                   linewidth=.5,
                                                   add_colorbar=False,
                                                    )
    # p = ftle.copy(data=gaussian(ftle.values, sigma=2)).plot.contourf(levels=21, cmap=cmr.gem, ax=ax, vmin=0,
    #                                                linewidth=.5,
    #                                                add_colorbar=False,
    #                                                 )
    # gradient.isel(elements=1).plot.contour(levels=11, cmap=cmr.pride, ax=ax,
    #                                                linewidth=.001,
    #                                                add_colorbar=True,
    #                                                 )
    p2 = flux_along_ridge.where(ridges>0).plot.contourf(levels=11,add_colorbar=False, cmap=cmr.amber, ax=ax, vmax=1000, vmin=200,
                                          )
    ridges_12.plot(add_colorbar=False, cmap=cmr.freeze)

    cbar_kwargs2 = {'label': r'Moisture flux along ridge ($\frac{Kg}{ms}$)',
                   'orientation': 'horizontal', 'shrink': .6, 'pad':0.02}
    cbar_kwargs = {'label': r'Moisture flux magnitude ($\frac{Kg}{ms}$)',
                   'shrink': 0.8,  'pad': 0.02}
    fig.colorbar(p2, ax=ax, **cbar_kwargs2)
    fig.colorbar(p, ax=ax, **cbar_kwargs)
    s = ax.quiver(uwind.longitude.values, uwind.latitude.values,
                  uwind.values, vwind.values, color='gray')

    # s = ax.quiver(u_vec.longitude.values, u_vec.latitude.values,
    #               u_vec.values, v_vec.values, color='gray')
    #
    # s = ax.quiver(gradient.longitude.values, gradient.latitude.values,
    #               gradient.isel(elements=0).values, gradient.isel(elements=1).values,
    #               color='red')

    qk = ax.quiverkey(s, 0.75, 0.90, 10, r'$10 \frac{m}{s}$', labelpos='E',
                          coordinates='figure')
    ax.coastlines(color='black')
    total_rain = np.sum(pr_)
    czs_rain = np.sum(ridges_ * pr_)
    local_strain_rain = np.sum(local_strain * pr_)
    rest = total_rain - local_strain_rain - czs_rain
        # ax.text(-90, -35, str(np.round(czs_rain.values / 1000)) + ' m rain on CZs \n ' +
        #             str(np.round(local_strain_rain.values / 1000)) + ' m rain on LStr \n' +
        #             str(np.round(rest.values / 1000)) + 'm remaining \n',
        #             bbox={'color': 'black', 'alpha': .2},
        #             color='black')
    ax.set_title('')
    ax.set_title(pd.Timestamp(flux_along_ridge.time.values).strftime('%Y-%m-%d %H:00') + ' UTC')

    plt.savefig(f'case_studies/{casestudy}/fig_flux_mag_{dt:02d}_area.png', dpi=600,
                    transparent=False, pad_inches=.2, bbox_inches='tight'
                    )

    plt.close()


    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    p = tcwv_.plot(cmap=cmr.gem, ax=ax, vmax=70, vmin=20,
                                                   linewidth=.5,
                                                   add_colorbar=False, zorder=0)
    # cbar = plt.colorbar(p,ax=ax, orientation='vertical',)
    # cbar.ax.set_ylabel( r'$V_{\rho_v}$ [$\frac{m}{s}$]')
    ridges.plot( add_colorbar=False, cmap=cmr.sunburst, zorder=10)
    ridges_12.plot(add_colorbar=False, cmap=cmr.freeze)

    contour = pr_.plot.contour(add_colorbar=False, levels=[1, 10],
                               alpha=.7, color='k', ax=ax, cmap=cmr.ocean_r, zorder=5)


    s = ax.quiver(uwind.longitude.values, uwind.latitude.values,
                  uwind.values, vwind.values, color='gray')

    ax.clabel(contour, fmt='%2.0f', colors='k', fontsize=10, use_clabeltext=True)
    qk = ax.quiverkey(s, 0.75, 0.92, 10, r'$10 \frac{m}{s}$', labelpos='E',
                          coordinates='figure')
    ax.coastlines(color='black')

    plt.savefig(f'case_studies/{casestudy}/fig_tcwv_{dt:02d}_area.png', dpi=600,
                    transparent=False, pad_inches=.2, bbox_inches='tight'
                    )
    plt.close()









def plot_chaotic_mixing():
    #TODO organise
    font = {'size': 4}

    matplotlib.rc('font', **font)
    x_departure = x_departure.interp(latitude=ridges.latitude.values,
                                           longitude=ridges.longitude.values,
                                           method='linear')
    y_departure = y_departure.interp(latitude=ridges.latitude.values,
            longitude=ridges.longitude.values,
            method='linear')
    fig = plt.figure(frameon=False)
    spec = GS(3, 1, figure=fig)
    axs=[]
    axs.append(fig.add_subplot(spec[0], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[1], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[2], projection=ccrs.PlateCarree()))

    x_p = x_departure.plot(transform=ccrs.PlateCarree(), ax=axs[0], add_colorbar=False,
                           cmap=cmr.fusion, vmin=-170,
                     vmax=x_departure.max(), antialiased=True, rasterized=True)
    y_p = y_departure.plot(transform=ccrs.PlateCarree(), ax=axs[1], add_colorbar=False, cmap=cmr.fusion_r,
                     vmin=y_departure.min(), vmax=y_departure.max(), antialiased=True, rasterized=True)
    p = ftle.plot(transform=ccrs.PlateCarree(), ax=axs[2], add_colorbar=False, cmap=cmr.freeze,
                   antialiased=True, rasterized=True, vmin=0.5, vmax=2.5)
    ridges.plot(ax=axs[2], cmap=cmr.sunburst, alpha=.8, transform=ccrs.PlateCarree(),
                add_colorbar=False, edgecolors='none')

    cbar = fig.colorbar(p, ax=axs[2], anchor = (0, .8), pad=0.005,
                        orientation='vertical')
    cbar1 = fig.colorbar(x_p, ax=axs[0], anchor = (0, .8),pad=0.005,
                         orientation='vertical')
    cbar2 = fig.colorbar(y_p, ax=axs[1], orientation='vertical', anchor = (0, 1),pad=0.005,
                         spacing='proportional')
    gl = axs[0].gridlines(draw_labels=True)

    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.xlines = False
    gl.ylines = False

    gl = axs[2].gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right = False
    gl.xlocator = ticker.FixedLocator([-80, -60, -40])
    gl.xlines = False
    gl.ylines = False

    cbar.ax.set_ylabel(r'FTLE $day^{-1}$')
    cbar1.ax.set_ylabel(r'Departure Lon.')
    cbar2.ax.set_ylabel(r'Departure Lat.')
    axs[2].coastlines(color='gray')
    axs[1].coastlines(color='gray')
    axs[0].coastlines(color='gray')
    axs[2].set_title(' ',)
    axs[1].set_title(' ')
    axs[0].set_title(' ')
    axs[2].set_title('c', loc='left')
    axs[1].set_title('b', loc='left')
    axs[0].set_title('a', loc='left')
    # plt.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.06)
    plt.savefig(f'case_studies/{casestudy}/chaotic_mixing_{dt:02d}.png',
                dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    time2 = time.time()
    ellapsed = time2 - time1
    print('Ellapsed time: ' + str(np.round(ellapsed / 60)) + 'minutes')



dt = 17
for s in np.arange(1.5e5, 1.5e6, 270000):
    time1 = time.time()
    timeseq = np.arange(0, 8) + dt
    ds = xr.merge([u, v])
    ds = ds.isel(time=timeseq)
    lcs = LCS(timestep=-6 * 3600, timedim='time', SETTLS_order=4, subdomain=subdomain, gauss_sigma=None, return_dpts=True)
    ftle, x_departure, y_departure = lcs(ds, s=s,) #resample='3H')
    ftle = np.log(ftle) / 2
    ftle = ftle.isel(time=0)

    fig = plt.figure(frameon=False)
    spec = GS(3, 1, figure=fig)
    axs=[]
    axs.append(fig.add_subplot(spec[0], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[1], projection=ccrs.PlateCarree()))
    axs.append(fig.add_subplot(spec[2], projection=ccrs.PlateCarree()))

    x_p = x_departure.plot(transform=ccrs.PlateCarree(), ax=axs[0], add_colorbar=False,
                           cmap=cmr.fusion, vmin=-170,
                     vmax=x_departure.max(), antialiased=True, rasterized=True)
    y_p = y_departure.plot(transform=ccrs.PlateCarree(), ax=axs[1], add_colorbar=False, cmap=cmr.fusion_r,
                     vmin=y_departure.min(), vmax=y_departure.max(), antialiased=True, rasterized=True)
    p = ftle.plot(transform=ccrs.PlateCarree(), ax=axs[2], add_colorbar=False, cmap=cmr.freeze,
                   antialiased=True, rasterized=True, vmin=0., vmax=2.5)


    cbar = fig.colorbar(p, ax=axs[2], anchor = (0, .8), pad=0.005,
                        orientation='vertical')
    cbar1 = fig.colorbar(x_p, ax=axs[0], anchor = (0, .8),pad=0.005,
                         orientation='vertical')
    cbar2 = fig.colorbar(y_p, ax=axs[1], orientation='vertical', anchor = (0, 1),pad=0.005,
                         spacing='proportional')
    gl = axs[0].gridlines(draw_labels=True)

    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_right = False
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.xlines = False
    gl.ylines = False

    gl = axs[2].gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right = False
    gl.xlocator = ticker.FixedLocator([-80, -60, -40])
    gl.xlines = False
    gl.ylines = False

    cbar.ax.set_ylabel(r'FTLE $day^{-1}$')
    cbar1.ax.set_ylabel(r'Departure Lon.')
    cbar2.ax.set_ylabel(r'Departure Lat.')
    axs[2].coastlines(color='gray')
    axs[1].coastlines(color='gray')
    axs[0].coastlines(color='gray')
    axs[2].set_title(' ',)
    axs[1].set_title(' ')
    axs[0].set_title(' ')
    axs[2].set_title('c', loc='left')
    axs[1].set_title('b', loc='left')
    axs[0].set_title('a', loc='left')
    # plt.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.06)
    plt.savefig(f'case_studies/{casestudy}/chaotic_mixing_{dt:02d}_s_{s}.png',
                dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

    time2 = time.time()
    ellapsed = time2 - time1
    print('Ellapsed time: ' + str(np.round(ellapsed / 60)) + 'minutes')


