import xarray as xr
import matplotlib as mpl
from convlib.diagnostics import apply_binary_mask
import sys
mpl.use('Agg')
from convlib.xr_tools import read_nc_files, createDomains
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import meteomath
import numpy as np
from convlib.diagnostics import add_basin_coord
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from convlib.xr_tools import xy_to_latlon
import cartopy.feature as cfeatur
from xrtools import xrumap as xru
import pandas as pd
from numba import jit
import numba
import convlib.xr_tools as xrtools
import os
if __name__ == '__main__':
    region = 'SACZ_big'
    years = range(1995, 2005)
    lcstimelen = 16
    basin = 'Uruguai'
    season = 'DJF'

    basin_origin='amazon'
    MAG = xr.open_dataset('~/phdlib/convlib/data/xarray_mair_grid_basins.nc')
    mask = MAG[basin]
    departures = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_departuretimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years, season=season, set_date=True, lcstimelen=lcstimelen, binary_mask=mask)
    ftle_array = read_nc_files(region=region,
                               basepath='/group_workspaces/jasmin4/upscale/gmpp/convzones/',
                               filename='SL_repelling_{year}_lcstimelen' + f'_{lcstimelen}_v2.nc',
                               year_range=years, season=season, binary_mask=mask, lcstimelen=lcstimelen,
                               set_date=True)
    pr = read_nc_files(region=region,
                      basepath='/gws/nopw/j04/primavera1/observations/ERA5/',
                      filename='pr_ERA5_6hr_{year}010100-{year}123118.nc',
                      year_range=years, transformLon=True, reverseLat=True,
                      time_slice_for_each_year=slice(None, None), season=season, binary_mask=mask)

    ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)
    #ftle_array = xr.apply_ufunc(lambda x: np.log(x), ftle_array ** 0.5)
    MAG = MAG.rename({'lat': 'latitude', 'lon': 'longitude'})
    #masked, origins = apply_binary_mask(departures.y_departure.time.values.copy(),
    #                                    dep_lat=departures.y_departure.copy(), dep_lon=departures.x_departure.copy(),
    #                                    mask=MAG[basin_origin], reverse=False)
    pr = pr * 1000
    pr_ts = pr.stack(points=['latitude', 'longitude']).mean('points')
    pr_ts = pr_ts.rolling(time=lcstimelen).mean().dropna('time')
    pr_ts = pr_ts.sel(time=ftle_array.time)
    ftle_ts = ftle_array.stack(points=['latitude', 'longitude']).mean('points')
    #ftle_ts = ftle_ts.resample(time='1D').mean()
    #pr_ts = pr_ts.resample(time='1D').mean()
    plt.figure(figsize=[10, 8])
    plt.style.use('fivethirtyeight')
    plt.scatter(x=pr_ts.values, y=ftle_ts.values)
    plt.xlabel('Precip (mm/6h)')
    plt.ylabel('FTLE')
    plt.savefig(f'tempfigs/diagnostics/precip_{basin}_{season}_{lcstimelen}.png')
    plt.close()

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import regionmask
    from matplotlib import cm, rc
    from matplotlib.ticker import MultipleLocator
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression as LR

    pca = PCA(n_components=3)
    x = departures.x_departure.groupby('time').mean().values.flatten()
    #x = x[~np.isnan(x)]
    y = departures.y_departure.groupby('time').mean().values.flatten()
    #y = y[~np.isnan(y)]
    z = pr.groupby('time').mean().rolling(time=lcstimelen).mean().\
        dropna('time').sel(time=departures.time).values.flatten()


    scaler = MinMaxScaler()
    scaler = scaler.fit(X=np.stack([x, y, z], axis=1))
    pca_data = scaler.transform(X=np.stack([x, y, z], axis=1))

    pca.fit(pca_data)
    result = pd.DataFrame(pca.transform(pca_data), columns=['PCA%i' % i for i in range(3)])
    result = result.set_index(pr_ts.time.values)
    colors = departures.y_departure.where(departures.Tiete == 1).groupby('time').mean()
    mask_ = (colors > -20.0).values
    colors = colors.where(colors <= -20.0, 1)
    colors = colors.where(colors > -20.0, 0)
    lr1 = LR()
    lr2 = LR()
    lr1.fit(X=ftle_ts.values[mask_].reshape(-1, 1), y=result['PCA0'].values[mask_].reshape(-1, 1))
    lr2.fit(X=ftle_ts.values[~mask_].reshape(-1, 1), y=result['PCA0'].values[~mask_].reshape(-1, 1))
    predicted1 = lr1.predict(X=ftle_ts.values[mask_].reshape(-1, 1))
    predicted2 = lr2.predict(X=ftle_ts.values[~mask_].reshape(-1, 1))

    plt.figure(figsize=[10, 10])
    plt.scatter(ftle_ts, result['PCA0'].values, c=colors, cmap='RdBu', alpha=0.7, s=pr_ts.values)
    plt.plot(ftle_ts.values[mask_], predicted1, color=cm.RdBu(500), lw=1.8)
    plt.plot(ftle_ts.values[~mask_], predicted2, color=cm.RdBu(0), lw=1.8)
    plt.xlabel('FTLE')
    plt.ylabel('PC1')
    plt.legend(['Origin North', 'Origin South'])

    plt.savefig('tempfigs/diagnostics/LinReg.pdf')
    plt.close()
    LR1 = lr1.score(X=ftle_ts.values[mask_].reshape(-1, 1), y=result['PCA0'].values[mask_].reshape(-1, 1))
    LR2 = lr2.score(X=ftle_ts.values[~mask_].reshape(-1, 1), y=result['PCA0'].values[~mask_].reshape(-1, 1))
    print(f'R2 statistics for fit1 is {LR1} and for fit 2 is {LR2}')
    amazon_xs = MAG.amazon.longitude.where(MAG.amazon == 1).values[~np.isnan(MAG.amazon.longitude.where(MAG.amazon == 1).values)]
    amazon_ys = np.ones_like(amazon_xs)*(-15) #-10 is the latitude fixed
    Tiete_xs = MAG.Tiete.longitude.where(MAG.Tiete == 1).values[~np.isnan(MAG.Tiete.longitude.where(MAG.Tiete == 1).values)]
    Tiete_ys = np.ones_like(Tiete_xs)*(-23.5) #-10 is the latitude fixed
    pc_amazon_border1 = pca.transform(scaler.transform(
        np.stack([amazon_xs.flatten(), amazon_ys.flatten(), np.zeros_like(amazon_xs).flatten()], axis=1)))[:, 0]
    pc_amazon_border2 = pca.transform(scaler.transform(
        np.stack([amazon_xs.flatten(), amazon_ys.flatten(), np.zeros_like(amazon_xs).flatten()], axis=1)))[:, 1]
    pc_amazon_border3 = pca.transform(scaler.transform(
        np.stack([amazon_xs.flatten(), amazon_ys.flatten(), np.zeros_like(amazon_xs).flatten()], axis=1)))[:, 2]
    pc_Tiete_border1 = pca.transform(scaler.transform(
        np.stack([Tiete_xs.flatten(), Tiete_ys.flatten(), np.zeros_like(Tiete_xs).flatten()], axis=1)))[:, 0]
    pc_Tiete_border2 = pca.transform(scaler.transform(
        np.stack([Tiete_xs.flatten(), Tiete_ys.flatten(), np.zeros_like(Tiete_xs).flatten()], axis=1)))[:, 1]
    pc_Tiete_border3 = pca.transform(scaler.transform(
        np.stack([Tiete_xs.flatten(), Tiete_ys.flatten(), np.zeros_like(Tiete_xs).flatten()], axis=1)))[:, 2]

    plt.figure(figsize=[10, 10])
    plt.style.use('default')

    pca_plot = plt.scatter(result['PCA0'].values, result['PCA1'].values, c=ftle_ts.values, cmap="viridis", alpha=0.7,
                           s=40*pr_ts.values ** 1.8, edgecolors='gray')
    plt.plot(pc_amazon_border1, pc_amazon_border2, color='darkgreen', lw=1.2)
    plt.plot(pc_Tiete_border1, pc_Tiete_border2, color='darkblue', lw=1.2)

    plt.ylabel('PC2')
    plt.xlabel('PC1')
    plt.colorbar(pca_plot)
    plt.savefig(f'tempfigs/diagnostics/PCA_1_2_lcstimelen{lcstimelen}.pdf')
    plt.close()

    plt.figure(figsize=[10, 10])
    plt.style.use('default')

    pca_plot = plt.scatter(result['PCA0'].values, result['PCA2'].values, c=ftle_ts.values, cmap="rainbow", alpha=0.7,
                           s=30*pr_ts.values ** 1.6)
    plt.plot([pca.transform(scaler.transform(np.array([-55, -15, 1.4]).reshape(1, -1)))[0][0],
             pca.transform(scaler.transform(np.array([-60, -18, 1.4]).reshape(1, -1)))[0][2]], color='black')
    plt.plot(pc_amazon_border1, pc_amazon_border3, color='darkgreen', lw=1.2)
    plt.plot(pc_Tiete_border1, pc_Tiete_border3, color='darkblue', lw=1.2)
    plt.ylabel('PC3')
    plt.xlabel('PC1')
    plt.colorbar(pca_plot)
    plt.savefig('tempfigs/diagnostics/PCA_1_3.pdf')
    plt.close()


    plot_3d = False
    if plot_3d:
        rc('font',size=28)
        rc('font',family='serif')
        rc('axes',labelsize=32)

        colors = pr_ts.values
        fig = plt.figure(figsize=[25, 15])
        plt.style.use('default')

        ax = fig.add_subplot(111, projection='3d')
        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        #[t.set_va('center') for t in ax.get_zticklabels()]
        #[t.set_ha('left') for t in ax.get_zticklabels()]

        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.xaxis._axinfo['tick']['outward_factor'] = 0.8
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0.8
        ax.zaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0.8
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.zaxis.set_major_locator(MultipleLocator(0.5))

        # make simple, bare axis lines through space:
        p1 = scaler.inverse_transform(pca.inverse_transform([[min(result['PCA0']), 0, 0], [max(result['PCA0']), 0, 0]]))
        p2 = scaler.inverse_transform(pca.inverse_transform([[0, min(result['PCA1']), 0], [0, max(result['PCA1']), 0]]))
        p3 = scaler.inverse_transform(pca.inverse_transform([[0, 0, min(result['PCA2'])], [0, 0, max(result['PCA2'])]]))

        ax.plot(xs=p1[:, 0], ys=p1[:, 1], zs=p1[:, 2], color='black')
        ax.text(p1[0, 0], p1[0, 1], p1[0, 2], "PC1 = " + str(round(pca.explained_variance_ratio_[0]*100)) + "%")
        ax.plot(xs=p2[:, 0], ys=p2[:, 1], zs=p2[:, 2], color='black')
        ax.text(p2[0, 0], p2[0, 1], p2[0, 2], "PC2 = " + str(round(pca.explained_variance_ratio_[1]*100)) + "%")
        ax.plot(xs=p3[:, 0], ys=p3[:, 1], zs=p3[:, 2], color='black')
        ax.text(p3[0, 0], p3[0, 1], p3[0, 2], "PC3 = " + str(round(pca.explained_variance_ratio_[2]*100)) + "%")
        ax.plot(xs=p1[:, 0], ys=p1[:, 1], zs=[0, 0], color='purple')

        c = ax.scatter(x, y, z, alpha=0.7, cmap='Blues', c=colors, s=(colors**1.7)*20)
        fig.colorbar(c)
        #ax.plot(x, y, z, lw=0.3, linestyle='-', alpha=0.4,
        #                 color='white')

        ax.grid(color='black')
        x, y = np.meshgrid(MAG['amazon'].longitude.values, MAG['amazon'].latitude.values)

        ax.contour(x, y,
                MAG['amazon'].values-1,  levels=[-0.00009, 1.00001], linewidths=0.8)
        ax.contour(x, y,
                MAG['Tiete'].values-1, levels=[-0.00009, 1.00001], linewidths=0.8)
        countries = regionmask.defined_regions.natural_earth.countries_110
        mask = countries.mask(np.arange(-70, -20), np.arange(-45, 5), wrap_lon=False)
        #mask = countries.mask(MAG.rename({'latitude': 'lat','longitude':'lon'}))
        mask = mask.where(np.isnan(mask), 1).where(~np.isnan(mask), 0) - 1
        x, y = np.meshgrid(mask.lon.values, mask.lat.values)
        ax.contour(x, y, mask.values, colors='black', levels=[-0.00009,1.00001], linewidths=0.8, linestyles='solid')
        ax.set_zlim(0, 2)
        ax.set_xlim(-80,-30)
        ax.set_ylim(-45, 10)
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Precip (mm/6h)')
        for angle in range(0, 360, 2):
            ax.view_init(30, angle)
            plt.savefig('tempfigs/3dplots/3dplot_{angle}.png'.format(angle=str(angle).format(':02d')))
    else:
        rc('font',size=28)
        rc('font',family='serif')
        rc('axes',labelsize=32)

        colors = pr_ts.values
        fig = plt.figure(figsize=[20, 20])
        plt.style.use('default')

        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # make simple, bare axis lines through space:
        p1 = scaler.inverse_transform(pca.inverse_transform([[min(result['PCA0']), 0, 0], [max(result['PCA0']), 0, 0]]))
        p2 = scaler.inverse_transform(pca.inverse_transform([[0, min(result['PCA1']), 0], [0, max(result['PCA1']), 0]]))
        p3 = scaler.inverse_transform(pca.inverse_transform([[0, 0, min(result['PCA2'])], [0, 0, max(result['PCA2'])]]))

        ax.plot(p1[:, 0], p1[:, 1], color='black')
        ax.text(p1[0, 0], p1[0, 1], "PC1 = " + str(round(pca.explained_variance_ratio_[0]*100)) + "%")
        ax.plot(p2[:, 0], p2[:, 1],  color='black')
        ax.text(p2[0, 0], p2[0, 1], "PC2 = " + str(round(pca.explained_variance_ratio_[1]*100)) + "%")
        ax.plot(xs=p3[:, 0], ys=p3[:, 1], color='black')
        ax.text(p3[0, 0], p3[0, 1], "PC3 = " + str(round(pca.explained_variance_ratio_[2]*100)) + "%")

        c = ax.scatter(x, y, alpha=0.9, cmap='plasma', c=colors, s=(colors**1.7)*35,
                       transform=ccrs.PlateCarree(), edgecolors='black')
        fig.colorbar(c)
        #ax.plot(x, y, z, lw=0.3, linestyle='-', alpha=0.4,
        #                 color='white')

        ax.grid(color='black')
        x, y = np.meshgrid(MAG['amazon'].longitude.values, MAG['amazon'].latitude.values)

        ax.contour(x, y,
                MAG['amazon'].values-1,  levels=[-0.00009,1.00001], linewidths=0.8,
                   colors=['white'], transform=ccrs.PlateCarree())
        ax.contour(x, y,
                MAG['Tiete'].values-1, levels=[-0.00009,1.00001], linewidths=0.8,
                colors=['white'], transform=ccrs.PlateCarree())

        ax.coastlines(color='black')
        os.environ["CARTOPY_USER_BACKGROUNDS"] = "/home/users/gmpp/phdlib/convlib/aux_files/"
        ax.background_img(name='BM', resolution='high')
        ax.set_xlim(-80, -30)
        ax.set_ylim(-45, 10)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.savefig(f'tempfigs/2dplots/2dplot_lcstimelen{lcstimelen}.pdf')
        plt.close()