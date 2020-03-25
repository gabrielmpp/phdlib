import os  # Must be implemented before dask

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import dask
from multiprocessing.pool import ThreadPool
import matplotlib
dask.config.set(scheduler='threads')  # Global config
dask.config.set(pool=ThreadPool(20))  # To avoid omp error
import xarray as xr
import scipy.signal as signal
import xarray as xr
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from phdlib.utils.xrumap import autoencoder as xru

import cartopy.crs as ccrs
import cartopy.feature as cfeature

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
def magnify():
    return [
        dict(selector="th", props=[("font-size", "8pt")]),
        dict(selector="td", props=[("padding", "0em 0em")]),
        dict(selector="th:hover", props=[("font-size", "12pt")]),
        dict(
            selector="tr:hover td:hover",
            props=[("max-width", "200px"), ("font-size", "12pt")],
        ),
    ]

def write_to_html_file(
    df,
    title="",
    filename="out.html",
    mode="forecast",
    cmap="GnBu",
    color1="#d65f5f",
    color2="#3361CC",
    low=0,
    high=0,
    axis=0,
):
    """
    Write a Pandas dataframe to an HTML file with nice formatting.
    """
    print("*" * 20)
    cm = matplotlib.cm.get_cmap(cmap)
    result = """
    <style>

        h2 {
            margin-top:80px;
            text-align: center;
            font-family: Helvetica, Arial, sans-serif;
        }
        table {
            margin-left: 80px;
            margin-right: 80px;
            margin-bottom: 50px;
        }
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }
        th, td {
            padding: 5px;
            text-align: center;
            font-family: Helvetica, Arial, sans-serif;
            font-size: 90%;
        }
        table tbody tr:hover {
            background-color: #dddddd;
        }
        .wide {
            width: 60%;
        }

    </style>
    <div position:"absolute">
    """
    result += "<h2> %s </h2>\n" % title
    # result += df.to_html(classes='wide', escape=False)
    if mode == "forecast":
        result += (
            df.style.background_gradient(cmap=cm, axis=axis, low=low, high=high)
            .set_table_styles(magnify())
            .render()
        )
    elif mode == "error":
        s = df.style.bar(align="left", color=[color1, color2], vmax=50).set_table_styles(
            magnify()
        )

        result += s.render()
    result += """
    </div>
    """
    with open(filename, "w") as f:
        f.write(result)
    return None

def coarsen_global_data(da: xr.DataArray, coarsen_by: float) -> xr.DataArray:
    Nlats = da.lat.values.shape[0] / coarsen_by
    Nlons = da.lon.values.shape[0] / coarsen_by
    lats = np.linspace(da.lat.values.min(), da.lat.values.max(), int(Nlats))
    lons = np.linspace(da.lon.values.min(), da.lon.values.max(), int(Nlons))
    da = da.sortby('lat')
    da = da.sortby('lon')
    da = da.interp(lat=lats, lon=lons, method='linear')
    return da


def latlonsel(array, array_like=None, lat=None, lon=None, latname='lat', lonname='lon'):
    """
    Function to crop array based on lat and lon intervals given by slice or list.
    This function is able to crop across cyclic boundaries.

    :param array: xarray.Datarray
    :param lat: list or slice (min, max)
    :param lon: list or slice(min, max)
    :return: cropped array
    """
    assert latname in array.coords, f"Coord. {latname} not present in array"
    assert lonname in array.coords, f"Coord. {lonname} not present in array"

    if isinstance(array_like, xr.DataArray):
        lat1 = array[latname].min().values
        lat2 = array[latname].max().values
        lon1 = array[lonname].min().values
        lon2 = array[lonname].max().values
    elif isinstance(lat, slice):
        lat1 = lat.start
        lat2 = lat.stop
    elif isinstance(lat, list):
        lat1 = lat[0]
        lat2 = lat[1]
    if isinstance(lon, slice):
        lon1 = lon.start
        lon2 = lon.stop
    elif isinstance(lon, list):
        lon1 = lat[0]
        lon2 = lat[1]

    lonmask = (array[lonname] < lon2) & (array[lonname] > lon1)
    latmask = (array[latname] < lat2) & (array[latname] > lat1)
    array = array.where(lonmask, drop=True)
    array = array.where(latmask, drop=True)
    return array


def model_analysis():
    ds = xr.open_dataset('data/ds.nc')

    array = ds.P.resample(time='1D').sum()
    print(array)
    array = array.stack(dict(stacked=['coupling', 'A_amazon']))
    # array = dict(array.groupby('time.season'))['DJF']
    groups = list(array.groupby('stacked'))
    power_list = []
    for label, group in groups:
        freq, power = signal.periodogram(group.values, axis=0)
        power_list.append(xr.DataArray(power, dims=['freq'], coords=dict(freq=freq)))

    array = xr.concat(power_list, dim=array.stacked)
    array = array.unstack()
    array = xr.apply_ufunc(lambda x: np.log(x), array)
    array = array.assign_coords(freq=1 / array.freq).rename(dict(freq='period'))
    # array = array.coarsen(period=Z, boundary='trim').mean()
    array.sel(A_amazon=10, period=slice(15, 2)).plot.contourf(levels=30, vmin=-26, vmax=-20, cmap='plasma')
    plt.show()
    array.sel(A_amazon=7, period=slice(15, 2)).plot.contourf(levels=30, vmin=-26, vmax=-20, cmap='plasma')
    plt.show()
    (array.sel(A_amazon=10, period=slice(15, 2)) - array.sel(A_amazon=7, period=slice(15, 2))).plot.contourf(levels=30,
                                                                                                             vmin=-3.5,
                                                                                                             vmax=3.5,
                                                                                                             cmap='RdBu')
    plt.show()
    array.sel(A_amazon=10, period=4, method='nearest').plot.line(x='coupling')
    plt.show()

    # qqplot

    ds = xr.open_dataset('data/ds.nc')
    idxs = np.array([pd.to_datetime(t).month in [1, 2, 12] for t in ds.time.values])
    ds = ds.sel(time=idxs)
    array = ds.P.resample(time='1D').sum() * 86400
    arrayQ = ds.Q.resample(time='1D').sum()  # * 86400
    array_CZ = array.where(arrayQ > arrayQ.quantile(0.8), drop=True)
    quants = np.arange(0, 1, 0.02)
    quant_cz = []
    quant_nocz = []
    for quant in quants:
        quant_cz.append(array_CZ.sel(coupling=4).quantile(quant).values)
        quant_nocz.append(array.sel(coupling=4).quantile(quant).values)

    plt.style.use('seaborn')
    plt.plot([0, 8], [0, 8], color='black', linestyle='dashed')

    plt.scatter(y=quant_cz, x=quant_nocz)
    plt.show()


def assert_time_is_last_entry(da):
    dims = list(da.dims)
    dims.remove('time')
    dims.append('time')
    return da.transpose(*dims)


def hilbert_spectrum(emd_array, out_type='period', apply_gauss=False):
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter
    def run_hilbert(group):
        analytic_signal = hilbert(group)
        phase = np.unwrap(np.angle(analytic_signal))
        #  freq = 4 * np.diff(phase) / (2 * np.pi)
        return phase

    # dask_array = emd_array.chunk(dict(encoded_dims=1, lat=1, lon=1))
    # temp_array = dask_array.stack({'stacked': ['encoded_dims', 'lat', 'lon']})
    emd_array = assert_time_is_last_entry(emd_array)
    phase = xr.apply_ufunc(lambda x: run_hilbert(x),
                           emd_array, dask='allowed'
                           )

    freq = phase.diff('time') / (2 * np.pi)
    freq = freq.unstack().load()

    if out_type == 'freq':
        return freq
    elif out_type == 'period':
        # periods_array = np.linspace(20, 4, 30)
        # periods_array = np.array([30, 25, 20, 15, 10, 5, 1])
        periods_array = np.array([2000, 90, 30, 15, 4, 0.5])
        k = 0
        periods = []
        emd_stacked = emd_array.stack({'points': ['lat', 'lon']}).isel(time=slice(None, -1))
        while k < (periods_array.shape[0] - 1):  # TODO make it faster
            mask = (freq > periods_array[k] ** -1) & (freq < periods_array[k + 1] ** -1)
            mask = mask.stack({'points': ['lat', 'lon']})
            periods.append(emd_stacked.where(mask).sum('encoded_dims').unstack())
            k += 1
            print(k)

        from scipy.ndimage import gaussian_filter

        array = xr.concat(periods, dim=pd.Index(periods_array[:-1], name='Period'))
        return array
    else:
        raise ValueError(f'out_type {out_type} not supported.')


def plot_periods(periods):
    # periods = periods.resample(time='M').sum('time')

    # periods_list = [5, 10, 15, 20, 25]
    periods_list = [90, 30, 15, 8, 4, 2]

    for time in periods.time.values:
        p = periods.sel(Period=periods_list, method='nearest').sel(time=time).plot.contourf(cmap='RdBu', vmax=10,
                                                                                            vmin=-10,
                                                                                            transform=ccrs.PlateCarree(),
                                                                                            col='Period', col_wrap=3,
                                                                                            aspect=periods.lon.shape[
                                                                                                       0] /
                                                                                                   periods.lat.shape[0],
                                                                                            levels=50, subplot_kws={
                'projection': ccrs.PlateCarree()})
        for i, ax in enumerate(p.axes.flat):
            fraction_of_power = np.abs(periods.sel(Period=periods_list[i],
                                                   method='nearest').sel(time=time)).sum(['lat', 'lon']).values / \
                                np.abs(periods.sel(time=time)).sum(['Period', 'lat', 'lon']).values
            ax.text(-44, -26, str(round(fraction_of_power * 100)) + '%')
            ax.coastlines()
            ax.add_feature(states_provinces, edgecolor='gray')
        plt.suptitle(pd.Timestamp(time).strftime("%Y-%m-%d"))
        plt.savefig(f'figs/plot_{time}.png')
        plt.close()

sp = dict(lat=-23, lon=-45, method='nearest')


regions = dict(
    sebr_coords=dict(lon=slice(-50, -40), lat=slice(-20, -30)),
    nbr_coords=dict(lon=slice(-70, -55), lat=slice(0, -15)),
    nebr_coords=dict(lon=slice(-45, -35), lat=slice(-3,-10)),
    sacz_coords=dict(lon=slice(-50, -45), lat=slice(-15, -20)),
)

# spc = xr.open_dataarray('/home/gab/phd/data/ds_emd.nc')


# ---- Line plots for SP ----#
# spc = xr.open_dataarray('/home/gab/phd/data/ds_9emd.nc').sel(**sp)
# periods = xr.open_dataarray('/home/gab/phd/data/ds_periods_9emd.nc').sel(**sp)
# chuva_reconstruida = periods.sel(time=slice('2009','2015' )).sum('Period')
# chuva_total = spc.sel(time=slice('2009', '2015')).sum('encoded_dims')
# chuva_reconstruida = chuva_reconstruida.where(chuva_reconstruida > 0, 0)
# plt.style.use('seaborn')
# plt.scatter(chuva_reconstruida.values, chuva_total.values, alpha=0.3)
# plt.plot([0,110], [0,110], color='black')
# plt.xlabel('Reconstructed rainfall (mm/day)')
# plt.ylabel('Observed rainfall (mm/day)')
# plt.show()
# chuva_total.plot()
# chuva_reconstruida.plot()
# plt.ylabel('Rainfall (mm/day)')
# plt.legend(['Observed rainfall', 'Reconstructed rainfall'])
# plt.show()
# p = spc.plot(col='encoded_dims', col_wrap=2, lw=0.2, aspect=3)
# plt.savefig('figs/panel_emd.pdf')
# periods.plot(row='Period', aspect=3)
# ---- computing Hilbert ----- #
# spc = xr.open_dataarray('/home/users/gmpp/ds_emd.nc')
spc = xr.open_dataarray('/home/gab/phd/data/ds_9emd.nc')
spc = spc.sel(time=slice('2009', '2011'))


#
from mia_lib.plotlib.miaplot import plot
spc = spc.chunk(dict(lat=10, lon=10))
periods = hilbert_spectrum(spc.isel(encoded_dims=slice(None, -   1)),
                           out_type='period', apply_gauss=False)
periods = periods.where(~np.isnan(spc.isel(encoded_dims=0))).drop('encoded_dims')
# mean_period_seasons = periods.groupby('time.season').mean('time') ** -1
# mean_period_seasons.name='Mean period (days)'
mean_power_seasons = periods.groupby('time.season').var('time') ** 0.5
mean_power_seasons.name = 'Intra-seasonal variability of daily rainfall (mm/day)'
# mean_period_seasons=mean_period_seasons.rename({'encoded_dims': 'IMF'})
# mean_power_seasons=mean_power_seasons.rename({'encoded_dims': 'IMF'})
ds = xr.merge([mean_power_seasons])
# ds = xr.merge([mean_period_seasons, mean_power_seasons])
sebr = ds.sel(**regions['sebr_coords']).mean(['lat', 'lon'])
nbr = ds.sel(**regions['nbr_coords']).mean(['lat', 'lon'])
nebr = ds.sel(**regions['nebr_coords']).mean(['lat', 'lon'])
sacz = ds.sel(**regions['sacz_coords']).mean(['lat', 'lon'])



areas = xr.concat([sebr, nbr, nebr, sacz],dim=pd.Index(['SEBR', 'Amazon', 'NEBR', 'SACZ'], name='Region'))

df = areas.to_dataframe()
# df = df.reset_index().set_index(['IMF', 'season']).pivot(columns='Region')
df = df.round(decimals=1)
# df = df.astype(str)
df.style.background_gradient(cmap='RdBu', axis=0).set_table_styles(magnify())
def formatter(x):
    return "\\" + 'bold '  + str(x)
formatters = [formatter for _ in range(df['Mean period (days)'].shape[1])]
with open('figs/table.txt', 'w') as file:
    file.write(df['Mean period (days)'].to_latex(multicolumn=True))
plt.style.use('fivethirtyeight')
ax = df.plot.bar(figsize=[20,10], y='Mean period (days)', log='y')
ax.axhline(y=365, color='red')
ax.axhline(y=30, color='red')
plt.savefig('figs/periods.pdf', bbox_inches='tight')
plt.style.use('seaborn')
axes = df['Intra-seasonal variability of daily rainfall (mm/day)'].unstack('Region').unstack('season').plot.bar(color='black',subplots=True, layout=[4, 4], legend=False, title=None, sharey=True, sharex=True)
for ax in axes.flatten():
    ax.set_ylabel('mm/day')
    ax.set_xticklabels(['1-4', '4-15', '15-30', '30-90', '> 90'])
    ax.set_xlabel('Period (days)')

plt.savefig('figs/barplot.pdf', bbox_inches='tight'); plt.close()
write_to_html_file(df, axis=0, filename='figs/table.html', cmap='RdBu', mode='error')


from shapely.geometry import LineString
to_plot=mean_power_seasons.sel(Period=15)
to_plot = to_plot.where(to_plot>0)
f, ax = plt.subplots( subplot_kw={'projection': ccrs.PlateCarree()})
p = to_plot.where(to_plot>0).sel(season='DJF').plot(levels=np.arange(0, 10, 1), transform=ccrs.PlateCarree(),
   vmin=0, vmax=20, cmap='BrBG', ax=ax, cbar_kwargs=dict(label='mm/day') )
ax.coastlines()

for region in regions.keys():
    coords = regions[region]
    x = [coords['lon'].start,
         coords['lon'].start,
         coords['lon'].stop,
         coords['lon'].stop,
         coords['lon'].start,

         ]
    y = [coords['lat'].start,
         coords['lat'].stop,
         coords['lat'].stop,
         coords['lat'].start,
         coords['lat'].start,
         ]

    ax.plot(x, y, lw=1.2, color='black', linestyle=':' )

ax.set_title(None)
plt.savefig('figs/subseasonal_variability.pdf', bbox_inches='tight')
plt.close()




from mia_lib.miacore.xrumap import autoencoder
mean_period = mean_period.sel(lon=slice(-60,None), )
mean_period = mean_period.stack({'points': ['lat', 'lon']}).transpose('points', ...)

mean_period = mean_period.dropna('points')
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4).fit_predict(mean_period.sel(encoded_dims=[4,5,6]).values)

kmeans = mean_period.sel(encoded_dims=3).copy(data=kmeans)
kmeans=kmeans.unstack()
kmeans.sortby('lat').sortby('lon').plot(cmap='YlGn'); plt.show()
encoder = autoencoder(mode='k_means', alongwith=['encoded_dims'], dims_to_reduce=['lat', 'lon'])
encoder = encoder.fit(mean_period)

da = encoder.transform(mean_period)
da=da.sortby('lon')
periods.to_netcdf('/home/users/gmpp/ds_periods.nc')

periods = xr.open_dataarray('/home/users/gmpp/ds_periods.nc')
periods = xr.open_dataarray('/home/gab/phd/data/ds_periods_9emd.nc')
spc = xr.open_dataarray('/home/gab/phd/data/ds_9emd.nc')

periods = periods.chunk({'Period': 6})

stdev = (periods.where(periods>0).groupby('time.season').mean('time')) #var('time')**0.5)
# p = stdev.plot.contourf(col='Period',
#                         row='season', transform=ccrs.PlateCarree(),vmax=15,
#                         aspect=periods.lon.shape[0] / periods.lat.shape[0], cmap='nipy_spectral',
#                         levels=50, subplot_kws={'projection': ccrs.PlateCarree()})
# for i, ax in enumerate(p.axes.flat):
#
#     ax.coastlines()
#     ax.add_feature(states_provinces, edgecolor='gray')
#
# plt.savefig(f'/home/users/gmpp/panel_stdev.png'); plt.close()

stdev_prec = spc.sum('encoded_dims').groupby('time.season').mean('time')#var('time')**0.5
spc = None
periods = None
period1 = stdev.sel(Period=2).drop('Period')
period2 = stdev.sel(Period=4).drop('Period') + stdev.sel(Period=8).drop('Period')
period3 = stdev.sel(Period=15).drop('Period') + stdev.sel(Period=30).drop('Period') + stdev.sel(Period=90).drop('Period')
stdev_ = xr.concat([stdev_prec, period1, period2, period3], dim=pd.Index(['total', '2', '2 - 8', '8 - 90'], name='Period'))
# stdev_ = coarsen_global_data(stdev_, coarsen_by=2.)
stdev_ = stdev_.where(stdev_>0, np.nan)

p = stdev_.plot.contourf(col='Period',levels=np.arange(3, 12, 1),
                        row='season', transform=ccrs.PlateCarree(),vmax=12, vmin=3,
                        aspect=stdev_.lon.shape[0] / stdev_.lat.shape[0], cmap='nipy_spectral_r',
                         subplot_kws={'projection': ccrs.PlateCarree()},
                        cbar_kwargs=dict(orientation="right"))
for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
plt.show()
#     ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'/home/users/gmpp/panel_stdev2.pdf'); plt.close()


import seaborn as sns
levels = np.arange(3, 12, 0.5)
palette = sns.color_palette("nipy_spectral_r", levels.shape[0])
p = stdev_.sel(season='DJF').plot(col='Period', cmap='nipy_spectral_r', vmin=3, vmax=12,
                        col_wrap=2, transform=ccrs.PlateCarree(),
                        aspect=stdev_.lon.shape[0] / stdev_.lat.shape[0],
                         subplot_kws={'projection': ccrs.PlateCarree()})

for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
plt.savefig('figs/panel_period.pdf')
plt.close()
# stdev_ = coarsen_global_data(stdev_, coarsen_by=3)
p = stdev_.sel(season='DJF').plot.contour(col='Period',
                        col_wrap=2, transform=ccrs.PlateCarree(),levels=[6, 6.1],
                        aspect=stdev_.lon.shape[0] / stdev_.lat.shape[0],
                         subplot_kws={'projection': ccrs.PlateCarree()})

for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
plt.show()
#     ax.add_feature(states_provinces, edgecolor='gray')

plt.savefig(f'/home/users/gmpp/panel_stdev3.pdf'); plt.close()