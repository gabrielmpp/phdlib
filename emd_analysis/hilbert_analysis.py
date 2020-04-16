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
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.signal as signal
import xarray as xr
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import statsmodels.api as sm

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from phdlib.utils.xrumap import autoencoder as xru
from scipy.signal import hilbert
from sklearn.cluster import KMeans

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.linear_model import LinearRegression
from scipy import stats

regions = dict(
    sebr_coords=dict(lon=slice(-52, -40), lat=slice(-15, -30)),
    nbr_coords=dict(lon=slice(-70, -55), lat=slice(0, -15)),
    nebr_coords=dict(lon=slice(-45, -35), lat=slice(-3,-10)),
    sacz_coords=dict(lon=slice(-50, -45), lat=slice(-15, -20)),
)
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


class HilbertSpectrum:
    """
    Class to compute quantities related to the Hilber spectrum. Assumes that time is the last dimension.
    """

    @staticmethod
    def analytic_signal(imf_da, dask='allowed'):
        return xr.apply_ufunc(lambda x: hilbert(x), imf_da, dask=dask)

    @classmethod
    def amplitude(cls, imf_da, dask='allowed'):
        analytic_da = cls.analytic_signal(imf_da, dask=dask)
        return xr.apply_ufunc(lambda x: np.abs(x), analytic_da, dask=dask)

    @staticmethod
    def phase(analytic_da, dask='allowed'):
        return xr.apply_ufunc(lambda x: np.unwrap(np.angle(x)), analytic_da, dask=dask)

    @classmethod
    def frequency(cls, imf_da, dask='allowed', sampledimname='time', timeunit='D'):
        """

        Parameters
        ----------
        imf_da : xr.DataArray
                    Array containing IMFs
        dask : str
                    Whether dask is allowed. See xr.apply_ufunc documentation for options.
        sampledimname : str
        timeunit : str

        Returns : xr.DataArray
        -------

        """
        analytic_da = cls.analytic_signal(imf_da, dask=dask)
        phase = cls.phase(analytic_da, dask=dask)
        freq = phase.differentiate(sampledimname, datetime_unit=timeunit) / (2 * np.pi)
        return freq.where(freq > 0, 0)

    @classmethod
    def groupbyfreqbins(cls,
                        freq_da: xr.DataArray,
                        amplitude_da: int,
                        nbins: Optional[int] = None,
                        nmin: Optional[int]=3,
                        bins=None,
                        bintype='freq',
                        deepcopy=True,
                        imfdim: Optional[str]='encoded_dims'):
        """
        Parameters
        ----------
        freq_da : xr.DataArray
                    Array representing the instantaneous frequencies of each IMF
        amplitude_da : xr.DataArray
                    Array representing the instantaneous amplitude of each IMF
        nbins : int
                    Desired number of frequency bins
        nmin : int
                    minimum number of points for the derivative to be stable. See eq. 7.3 in Huang et al. (1999)
        imfdim : str
                    name of dimension containing IMF coords

        Returs : xr.DataArray
                    array with amplitude accumulated by frequency bins.
        """
        # dt = freq_da.time.diff('time').values[0]
        if deepcopy:
            freq_da = freq_da.copy()
        dt = 1 # days
        T = freq_da.time.shape[0]
        N = cls._assert_nbins(T, dt=1, n=nmin)
        if isinstance(nbins, type(None)):
            nbins = N
        elif nbins > N:
            nbins = N
        if bintype == 'period':
            freq_da = freq_da ** -1
        if not isinstance(bins, type(None)):
            freqmin = freq_da.min().values
            freqmax = freq_da.max().values
            bins = np.linspace(freqmin, freqmax, nbins)

        binszip = zip(bins[:-1], bins[1:])
        out_da = []
        for f0, f1 in binszip:
            print(f0, f1)
            mask = (freq_da > f0) & (freq_da < f1)
            out_da.append(amplitude_da.where(mask).sum(imfdim))
            print('Done {}'.format(str(f0)))
        out_da = xr.concat(out_da, dim=pd.Index(bins[1:], name=bintype))
        return out_da
    @staticmethod
    def _assert_nbins(T, dt, n=3):
        """
        Internal method to return a number of bins that attend the restriction
        from eq. 7.3 in Huang et al. (1999)
        Parameters
        ----------
        T
        dt
        n

        Returns
        -------

        """
        return int(T/(n*dt))


def hilbert_spectrum(emd_array, out_type='period', apply_gauss=False):
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter
    def run_hilbert(group, mode):
        analytic_signal = hilbert(group)
        amplitude=np.abs(analytic_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        #  freq = 4 * np.diff(phase) / (2 * np.pi)
        if mode=='amplitude':
            return amplitude
        elif mode=='phase':
            return phase

    # dask_array = emd_array.chunk(dict(encoded_dims=1, lat=1, lon=1))
    # temp_array = dask_array.stack({'stacked': ['encoded_dims', 'lat', 'lon']})
    emd_array = assert_time_is_last_entry(emd_array)
    phase = xr.apply_ufunc(lambda x: run_hilbert(x, mode='phase'),
                           emd_array, dask='allowed'
                           )
    amplitude = xr.apply_ufunc(lambda x: run_hilbert(x, mode='amplitude'),
                           emd_array, dask='allowed'
                           )

    freq = phase.differentiate('time', datetime_unit='D') / (2 * np.pi)
    freq = freq.where(freq > 0, 0)
    freqmin = freq.min()
    freqmax = freq.max()
    freqbins = np.linspace(freqmin, freqmax, 100)

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


spc = xr.open_dataarray('/home/gab/phd/data/ds_ceemdan.nc')
masks_dict = dict()

pseudo_imfs = spc.isel(encoded_dims=slice(0, -2)) # removing residual and noise


region_list=['sebr_coords', 'sacz_coords',]
fig, axs = plt.subplots(2, 2, sharey=True)

for region in region_list:
    peaks = xr.apply_ufunc(lambda x: len(find_peaks(x)[0]), pseudo_imfs.sel(**regions[region], encoded_dims=slice(0,9)), input_core_dims=[['time']], vectorize=True,
                            dask='allowed', output_dtypes=[np.float32])


    period = pseudo_imfs.time.shape[0]/peaks
    period = period.where(period != np.inf, np.nan)
    period_labels = [str(round(p,1) )for p in period.mean(['lat', 'lon']).values]
    period_labels_max = [str(round(p,1) )for p in period.max(['lat', 'lon']).values]
    period_labels_min = [str(round(p,1) )for p in period.min(['lat', 'lon']).values]
    energy_total = ((pseudo_imfs.sel(encoded_dims=slice(0,9), **regions[region]) ** 2)**0.5).mean('time')

    energy_season = ((pseudo_imfs.sel(encoded_dims=slice(0,9), **regions[region]) ** 2)**0.5).groupby('time.season').mean('time')
    tot_energy = energy_season.sum('encoded_dims').values
    energy_season['accumulated energy'] = ('encoded_dims', 'season', 'lat','lon',), \
                                          [energy_season.sel(encoded_dims=slice(0, ed)).sum('encoded_dims').values/tot_energy for ed in energy_season.encoded_dims.values]


    for i, season in enumerate(energy_season.season.values):
        ax = axs.flatten()[i]
        ax.plot(energy_season.encoded_dims.values, energy_season.sel(season=season).mean(['lat', 'lon']).values)
        ax.plot(energy_total.encoded_dims.values, energy_total.mean(['lat', 'lon']).values, color='black')
        ax.set_title(season)
        ax.set_xlabel('Pseudo-IMF from CEEMDAN')
        ax.set_ylabel('Mean energy (mm)')

plt.show()

peaks = xr.apply_ufunc(lambda x: len(find_peaks(x)[0]), pseudo_imfs.sel(encoded_dims=slice(0, 9)),
                       input_core_dims=[['time']], vectorize=True,
                       dask='allowed', output_dtypes=[np.float32])

period = pseudo_imfs.time.shape[0] / peaks
period = period.where(period != np.inf, np.nan)
pseudo_imfs['month'] = 'time', [pd.Timestamp(x).strftime('%m') for x in pseudo_imfs.time.values]
pseudo_imfs['year'] = 'time', [pd.Timestamp(x).strftime('%Y') for x in pseudo_imfs.time.values]

energy_total = ((pseudo_imfs.sel(encoded_dims=slice(0, 9)) ** 2) ** 0.5).groupby('time.season').mean('time')


for i, encoded_dim in enumerate(energy_total.encoded_dims.values):
    fig, axs = plt.subplots(1, 2, subplot_kw=dict(projection=ccrs.PlateCarree()))#, figsize=[14, 10])
    p1 = energy_total.isel(encoded_dims=encoded_dim).sel(season='DJF').plot.contourf(title=None,vmin=0, levels=9,vmax=8,ax=axs[1],add_label=True, cmap='BrBG', cbar_kwargs=dict(orientation='horizontal'))
    p2 = period.sel(encoded_dims=encoded_dim).plot.contourf(levels=14,title=None, cmap='RdBu', ax =axs[0],add_label=True, cbar_kwargs=dict(orientation='horizontal'))
    # period.sel(encoded_dims=encoded_dim).plot.contour(levels=[5,8, 15], cmap='Greys', ax =axs[0], add_colorbar=False)

    for ax in axs:
        ax.coastlines()
    axs[0].set_title('IMF ' + str (i) + ' average period (days)')
    axs[1].set_title('IMF ' + str (i) + ' average power (mm/day)')

    plt.tight_layout()
    plt.savefig(f'figs/imf_{i}.png')
    plt.close()


# ---- Cluster analysis ----- #



IMF_groups ={
    '0 - 7': slice(0, 7),
    # '2 - 3': slice(2, 3),
    # '4 - 5': slice(4, 5),

    # '6 - 9': slice(6, 9)
}

nclusterss = np.arange(1, 10, 1)
inertiass = dict.fromkeys(IMF_groups.keys())
for imf_group in IMF_groups.keys():
    edims = IMF_groups[imf_group]
    data_to_cluster = np.log(period.sel(encoded_dims=edims).stack({'points': ['lat', 'lon']}). \
        transpose('points', 'encoded_dims').dropna('points'))
    weights = np.log(energy_total.sel(season='DJF').stack({'points': ['lat', 'lon']}).dropna('points'). \
        sel(encoded_dims=edims).sum('encoded_dims').values)

    inertiass[imf_group] = []
    for nclusters in nclusterss:

        km = KMeans(n_clusters=nclusters)
        clusterized_data = data_to_cluster.isel(encoded_dims=0).drop('encoded_dims').\
        copy(data=km.fit_predict(data_to_cluster)).unstack().sortby('lat').sortby('lon').unstack()
        inertiass[imf_group].append(km.inertia_)

    inertiass[imf_group] = inertiass[imf_group] / inertiass[imf_group][0]

plt.style.use('seaborn')
keys = inertiass.keys()
for key in keys:
    inertias = inertiass[key]
    plt.plot(nclusterss, inertias)

plt.hlines([0.2], xmin=nclusterss[0], xmax=nclusterss[-1], linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances to the center of the clusters')
plt.savefig('figs/elbow_test.pdf')

# Now we determined 4 clusters for IMFS 0-1 and 3 clusters for the others

IMF_groups =[
    [slice(0, 7), 3], # second position of the list is the number of clusters required
     [slice(2, 3), 3],
     [slice(4, 5), 3],
    ]
clusterized_data = []
cluster_centers = []
for imf_group in IMF_groups:
    edims = imf_group[0]
    data_to_cluster = period.sel(encoded_dims=edims).stack({'points': ['lat', 'lon']}). \
        transpose('points', 'encoded_dims').dropna('points')
    weights = energy_total.sel(season='DJF').stack({'points': ['lat', 'lon']}).dropna('points'). \
        sel(encoded_dims=edims).sum('encoded_dims').values
    km = KMeans(n_clusters=imf_group[1])

    clusterized_data.append(
        data_to_cluster.isel(encoded_dims=0).drop('encoded_dims'). \
        copy(data=km.fit_predict(np.log(data_to_cluster), sample_weight=np.log(weights))).unstack().sortby('lat').sortby(
        'lon').unstack()
    )
    cluster_centers.append(km.cluster_centers_)

import matplotlib.ticker as ticker
def format_label(x, pos):
    if pos < cluster_centers[0].shape[0]:
        return np.round(cluster_centers[0][pos],1)
    else:
        return None
format=ticker.FuncFormatter(format_label)
clusterized_data = xr.concat(clusterized_data, dim=pd.Index(['0 - 7', '2 - 3', '4 - 5'], name='IMF_groups'))
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
p = clusterized_data.sel(IMF_groups='0 - 7').plot.contourf(hatches=['X','+'],levels=4,cmap='rainbow', ax=ax, transform=ccrs.PlateCarree(),
                                                      add_labels=True, add_colorbar=True)
    # c = clusterized_data.sel(IMF_groups='4 - 5').plot.contour(levels=6,cmap='black', ax=ax, transform=ccrs.PlateCarree())
    # plt.clabel(c, inline=1,fmt = '%1.0f')
    # cbar = fig.colorbar(p, format=format)
    # cbar.ax.set_yticklabels(np.round(cluster_centers[0][:,0], 1))  # horizontal colorbar
ax.coastlines()
plt.savefig('figs/regions_log.pdf')

edims1 = '0 - 7'

d1 = clusterized_data.sel(IMF_groups=edims1)

a = pseudo_imfs.where(d1 == 0).mean(['lat', 'lon'])#.shift(time=5).dropna('time')
b = pseudo_imfs.where(d1 == 2).mean([ 'lat', 'lon'])
b = b.sel(time=a.time)
a.name='B'
b.name='C'
ds = xr.merge([a, b])
ds.plot.scatter(x='C', y='B', hue='month', cmap='tab10', hue_style='discrete', col='encoded_dims', col_wrap=3)
plt.show()
b = b.sel(time=a.time)
for month in a.month:
    plt.scatter(a, b, alpha=0.2)
    aa = a.where(a.month == month, drop=True).values
    bb = b.where(b.month == month, drop=True).values
    lm = LinearRegression()
    y = lm.fit(aa.reshape(-1,1), bb.reshape(-1,1))
    plt.plot(aa, lm.predict(aa.reshape(-1,1)))

plt.legend(np.unique(a.month.values).tolist())

plt.show()
plt.style.use('seaborn')

a.groupby('time.month').var('time').plot.line(x='month')
b.groupby('time.month').var('time').plot.line(x='month')

plt.show()

cluster_centers[0][0]
cluster_centers[0][1]

from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.recursive_ls import RecursiveLS
colors = [float(x) for x in a.where(b.month == '01', drop=True).year.values]
Y = a.where(b.month == '01', drop=True).sel(encoded_dims=[3]).sum('encoded_dims').values
X = b.where(a.month == '01', drop=True).sel(encoded_dims=[2]).transpose('time', ...).values
X = sm.add_constant(X )
mod = sm.OLS(Y,X).fit()
from contextlib import redirect_stdout
with open('regression_2nd_IMF_C_3rd_IMF_B.txt', 'w') as file:
    with redirect_stdout(file):
        print(mod.summary2())
prstd, iv_l, iv_u = wls_prediction_std(mod, alpha=0.1)


plt.style.use('seaborn-deep')
fig, ax = plt.subplots(figsize=(8,6))

ax.scatter(X[:,1], Y, )
ax.plot(X[:,1], mod.fittedvalues, 'r', label="Linear regression")
ax.plot(X[:,1], iv_u, c='gray', linestyle='-' )
ax.plot(X[:,1], iv_l, c='gray', linestyle='-',label='Conficence band')
ax.legend(loc='best')
ax.set_ylabel('First IMF in region C (mm/day)')
ax.set_xlabel('First IMF in region A (mm/day)')
plt.savefig('figs/linear_regression_C_A.pdf')

p_values = fii.summary2().tables[1]['P>|t|']

plt.scatter(X, Y)
           # c=a.where(b.month == '01', drop=True).year, cmap='rainbow')
plt.legend()

#---- ROLLING OLS -----#
Y = a.sel(encoded_dims=3).to_dataframe(name='Y')
X = a.sel(encoded_dims=1).transpose('time', ...).shift(time=8).to_dataframe(name='X')

rols = RollingOLS(Y['Y'], X[['X']],window=35 )
rres = rols.fit()

params = rres.params
plt.style.use('seaborn')
fig = rres.plot_recursive_coefficient(figsize=(14,6))
plt.ylim([-1, 1])
plt.show()

X['X'].plot()
plt.show()
XX = sm.tsa.tsatools.lagmat(X['X'], maxlag=10)
rols = RollingOLS(Y['Y'], XX, window=60)
rres = rols.fit()
plt.style.use('seaborn')
fig = rres.plot_recursive_coefficient( figsize=(14,6))
plt.ylim([-1.5, 3])
plt.title(None)
plt.show()


lm = LinearRegression()
lm.fit(X,
       Y.values.reshape(-1,1))
y = lm.predict(X)
plt.plot(X[:, 0], y, color='black')
lm.coef_
lm.intercept_
lm.score(X,Y)
plt.show()
a.sel(encoded_dims=3, time='2011').plot()
b.sel(encoded_dims=3, time='2011').plot()
plt.legend(['Amazon', 'SEBR'])
plt.show()
plt.style.available

plt.scatter(b.sel(encoded_dims=3, time='2011'),
            a.sel(encoded_dims=3, time='2011'), alpha=0.7,
            c=a.sel(time='2011').month, cmap='rainbow')
plt.show()

lr  = LR()
lr.fit(x=b, y=a)

(a).sel(time='2012').plot()
b.sel(time='2012').plot()
c.sel(time='2010').plot()



#----- EOT -----#
def covariance_gufunc(x, y):
    return ((x - x.mean(axis=-1, ))
            * (y - y.mean(axis=-1,))).mean(axis=-1)

def pearson_correlation_gufunc(x, y):
    return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

from scipy import stats
N_points = 1
edim=3
time_series_original = pseudo_imfs.sel(encoded_dims=edim).where(d1==2).mean(['lat', 'lon'])
time_series = time_series_original.copy()

data = pseudo_imfs.sel(encoded_dims=edim).where(d1==2)
poss = []
for i in range(N_points):
    # ---- Finding best correlation ---- #
    pearson = pearson_correlation_gufunc(data.transpose(..., 'time'), time_series_original).unstack().sortby('lat').sortby('lon')
    cov = covariance_gufunc(data.transpose(..., 'time'), time_series).unstack().sortby('lat').sortby('lon')
    max_pos = pearson[np.unravel_index(np.argmax(pearson**2), pearson.shape)]
    max_pos.name = str(i)
    # ---- Training linear model and subtracting from data ---- #
    lm = LinearRegression()
    Y = data.stack({'points': ['lat', 'lon']}).dropna('points')
    X = time_series_original.values
    lm.fit(X.reshape([-1,1]), Y)
    preds = Y.copy(data=lm.predict(X.reshape([-1, 1]))).unstack().sortby('lat').sortby('lon')
    # ---- Residual ---- #
    time_series = data.sel(lat=max_pos.lat, lon=max_pos.lon).drop('lat').drop('lon')
    data = data - preds

    # ---- Plotting ---- #
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
    pearson.plot(ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(max_pos.lon.values, max_pos.lat.values)
    poss.append(max_pos)
    plt.show()





poss = xr.concat(poss, dim=pd.Index(np.arange(N_points),name='index'))
time_series_amazon_point1_edim_2 = time_series
time_series_sacz_point2_edim_2 = time_series
time_series_amazon=time_series
time_series.plot()
plt.show()





#---- ROLLING OLS -----#
Y = time_series_sacz_point2_edim_2.to_dataframe(name='Y')
X = time_series_amazon_point1_edim_2.to_dataframe(name='X')

rols = RollingOLS(Y['Y'], X[['X']],window=60 )
rres = rols.fit()

params = rres.params
plt.style.use('seaborn')
fig = rres.plot_recursive_coefficient(figsize=(14,6))
# plt.ylim()
plt.show()




np.correlate(stacked, time_series)
cov_matrix = stats.pearsonr(stacked.values, y=time_series.values)
cov_matrix = cov_matrix[np.triu_indices(cov_matrix.shape[0])]
stacked
cov_matrix.flatten().shape
cov_matrix = xr.DataArray(cov_matrix, dims=[''])

np.diag(cov_matrix).shape
time_series.plot()
plt.show()



# energy_total = ((pseudo_imfs.sel(encoded_dims=slice(0, 9)) ** 2) ** 0.5).mean('time') / ((pseudo_imfs.sel(encoded_dims=slice(0, 9)) ** 2) ** 0.5).mean('time').sum('encoded_dims')








spc['month'] = ('time'), [pd.Timestamp(x).strftime('%m') for x in spc.time.values]

amazon_subseasonal = spc.sel(regions['nbr_coords']).sel(encoded_dims=3).where(spc.month=='01', drop=True).mean(['lat', 'lon']).values
sacz_subseasonal = spc.sel(regions['sebr_coords']).sel(encoded_dims=3).where(spc.month=='01', drop=True).mean(['lat', 'lon']).values
plt.plot(amazon_subseasonal)
plt.plot(sacz_subseasonal)
plt.show()
lr = LinearRegression()
lr.fit(amazon_subseasonal.reshape(-1,1), sacz_subseasonal.reshape(-1,1))
predicted = lr.predict(amazon_subseasonal.reshape(-1,1))
plt.scatter(amazon_subseasonal, sacz_subseasonal)
plt.plot(amazon_subseasonal, predicted)
plt.show()



p=energy_season.sel(season='DJF').plot.contourf(levels=21, col='encoded_dims', col_wrap=3, vmax=8,cmap='BrBG',
                                                subplot_kws={'projection': ccrs.PlateCarree()})
for i, ax in enumerate(p.axes.flat):
    ax.coastlines()
    ax.set_title(period_labels_min[i] + '-' + period_labels[i] + '-' + period_labels_max[i])
plt.show()





# energy_density
error_bar = np.sqrt(2*period/pseudo_imfs.time.shape[0])

plt.bar(height=energy_density, x=np.arange(0,9,1), yerr=error_bar)
plt.xticks(np.arange(0,9,1), period_labels)
plt.ylabel('Energy density scaled')
plt.xlabel('Average period (days)')
plt.title('Mean energy density and theoretical standard deviation\n for each pseudo-IMF in the city of Sao Paulo')
plt.savefig('energy_sensity_sp.pdf')

((pseudo_imfs)/(energy_residual)).isel(encoded_dims=5).plot.hist()
plt.show()
confidence1 = - np.log(period.values) + 0.675*np.sqrt(2/pseudo_imfs.time.shape[0])*np.exp(np.log(period.values)/2)
confidence2 = - np.log(period.values) - 0.675*np.sqrt(2/pseudo_imfs.time.shape[0])*np.exp(np.log(period.values)/2)
plt.scatter(x=np.log(period.values), y=np.log(energy_density.values))
plt.plot( np.log(period.values), confidence1, color='black', linestyle='--')
plt.plot( np.log(period.values), confidence2, color='black',linestyle='--')
plt.xlabel('Log of the average periods')
plt.ylabel('Log of the energy density scaled')
# plt.loglog()
plt.title('Statistical significance of the pseudo IMFs')
plt.savefig('figs/statistical.pdf')
plt.close()
# ---- Plotting Pseudo IMFs ---- #
plt.style.use('seaborn')
pseudo_imfs.plot.line(x='time', row='encoded_dims',aspect=5, sharey=True)

plt.show()


plt.plot(energy_density)


spc = spc.sel(**regions['sacz_coords'])
# spc = xr.apply_ufunc(lambda x, y: x - y, spc.groupby('time.month'), spc.groupby('time.month').mean('time')) # anomaly
energy = ((spc.sel(encoded_dims=1))**2).sum()/spc.time.shape[0]
f_spc = xr.apply_ufunc(lambda x: np.abs(np.fft.fft(x, axis=2)), pseudo_imfs.sel(encoded_dims=slice(0,4)))
freqs = np.fft.fftfreq(spc.time.shape[0])
f_spc = f_spc.assign_coords(time=(freqs**-1)).rename({'time': 'logperiod'}).sortby('logperiod').isel(logperiod=slice(None,-1))
(f_spc).sel( logperiod=slice(None, None),).where(f_spc.logperiod>0, drop=True).plot.line(hue='encoded_dims',x='logperiod')
plt.loglog()
plt.xlabel('Log of the period')
plt.savefig('figs/_ceemdan.pdf')
plt.close()
f_spc = f_spc.rename
f_spc = f_spc.mean(['lat', 'lon'])
f_spc.plot.line(x='time')

spc = spc.chunk(dict(encoded_dims=spc.encoded_dims.values.shape[0]))
spc=spc.isel(time=slice(None, 365))
spc = assert_time_is_last_entry(spc)
spc = spc.isel(encoded_dims=slice(None, -1))
freq = HilbertSpectrum.frequency(spc)
amplitude_da = HilbertSpectrum.amplitude(spc)
freq_grouped = HilbertSpectrum.groupbyfreqbins(freq, amplitude_da, nbins=30)
freq_grouped = freq_grouped.where(spc.isel(time=0, encoded_dims=0) != np.nan,  np.nan)
freq_grouped = freq_grouped.drop('encoded_dims')
amplitude_grouped_by_periods = freq_grouped.assign_coords({'freq': freq_grouped.freq ** -1}).rename({'freq': 'period'})
amplitude_grouped_by_periods.sel(**regions['sacz_coords']).mean(['lat', 'lon']).plot.contourf(levels=50,x='time', cmap='nipy_spectral')
plt.show()
freq_grouped.sel(**regions['nbr_coords']).mean(['lat', 'lon']).plot.contourf(levels=50,ylim=[5/365, 1/5],
                                                                              x='time', cmap='nipy_spectral')
plt.show()
marginal_spectrum_sacz = freq_grouped.sel(**regions['sacz_coords']).mean(['lat', 'lon']).sum('time').values
marginal_spectrum_nbr = freq_grouped.sel(**regions['nbr_coords']).mean(['lat', 'lon']).sum('time').values
marginal_spectrum_nebr = freq_grouped.sel(**regions['nebr_coords']).mean(['lat', 'lon']).sum('time').values
marginal_spectrum_sebr = freq_grouped.sel(**regions['sebr_coords']).mean(['lat', 'lon']).sum('time').values

plt.scatter(y=marginal_spectrum_sacz, x=freq_grouped.freq.values, marker='D')
plt.scatter(y=marginal_spectrum_nbr, x=freq_grouped.freq.values, marker='D')
plt.scatter(y=marginal_spectrum_nebr, x=freq_grouped.freq.values, marker='D')
plt.scatter(y=marginal_spectrum_sebr, x=freq_grouped.freq.values, marker='D')
plt.legend(['sacz', 'nbr', 'nebr', 'sebr'])
plt.semilogx(True)
plt.semilogy(True)
plt.show()
marginal_spectrum_sacz = freq_grouped.sel(**regions['sacz_coords']).mean(['lat', 'lon']).sum('freq').values
marginal_spectrum_nbr = freq_grouped.sel(**regions['nbr_coords']).mean(['lat', 'lon']).sum('freq').values
marginal_spectrum_nebr = freq_grouped.sel(**regions['nebr_coords']).mean(['lat', 'lon']).sum('freq').values
marginal_spectrum_sebr = freq_grouped.sel(**regions['sebr_coords']).mean(['lat', 'lon']).sum('freq').values

plt.scatter(y=marginal_spectrum_sacz, x=freq_grouped.time, marker='D')
plt.scatter(y=marginal_spectrum_nbr, x=freq_grouped.time, marker='D')
plt.scatter(y=marginal_spectrum_nebr, x=freq_grouped.time, marker='D')
plt.scatter(y=marginal_spectrum_sebr, x=freq_grouped.time, marker='D')
plt.legend(['sacz', 'nbr', 'nebr', 'sebr'])

plt.show()

sacz = xr.open_dataarray('~/da_amplitude_sacz.nc')
nbr = xr.open_dataarray('~/da_amplitude_nbr.nc')
nebr = xr.open_dataarray('~/da_amplitude_nebr.nc')
sebr = xr.open_dataarray('~/da_amplitude_sebr.nc')

da = xr.concat([sacz, nbr, nebr, sebr], pd.Index(['sacz', 'nbr', 'nebr', 'sebr'], name='region'))

#
# period_timedeltas = [pd.Timedelta(x, unit='D') for x in np.diff(da.period.values)]
# period_timedeltas.append(period_timedeltas[-1] )
# da['period_timedeltas'] = 'period', period_timedeltas
# da = da.assign_coords(period=period_timedeltas)
# da=da.sortby('period')
# da
# da = da.resample(period='20D').mean(skipna=True)
# daa = da.groupby('time.season').mean('time')
# da.plot( row='region')
# plt.show()

from scipy.signal import resample
daa = resample(da.values, da.period.shape[0], axis=1) # resampling periods
new_periods = np.linspace(da.period.min().values, da.period.max().values,  da.period.shape[0],endpoint=False)
daa = xr.DataArray(daa, dims=da.dims, coords=dict(
    region=da.region,
    period=new_periods,
    time=da.time
))
bins = np.round(np.linspace(5, 18, 5))
bins_zip = zip(bins[0:-1], bins[1:])
arr_list = []
for b1, b2 in bins_zip:
    mask = (daa.period > b1) & (daa.period < b2)
    arr_list.append(daa.where(mask, drop=True).sum('period'))

arr_binned = xr.concat(arr_list, dim=pd.Index(bins[1:], name='period_bins'))
df = arr_binned.groupby('time.season').mean('time').to_dataframe(name='a').unstack('region').unstack('season')
df.plot.bar(subplots=True, layout=[4, 4],  legend=False, title=None,sharey=False,  sharex=True)
plt.show()


daa.plot(col='region', col_wrap=2, vmin=0, cmap='nipy_spectral', cbar_kwargs=dict(orientation='h'))
plt.show()
daa.sum('time').plot.line(hue='region')
plt.loglog(True)
plt.show()



period_timedeltas = [pd.Timestamp(x, unit='D') for x in -np.diff(da.period.values)]
period_timedeltas.append(period_timedeltas[-1] + pd.Timedelta(1))
da = da.assign_coords(period=period_timedeltas)
da=da.sortby('period')
da = da.resample(period='1D').sum(skipna=False)
da.plot(col='region')
plt.show()
da
da.period
daa = daa.resample(period=1)
daa.plot.line(col='season',col_wrap=2,hue='region')
plt.show()
# ---- L
# ine plots for SP ----#
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
# spc = xr.open_dataarray('/home/gab/phd/data/ds_9emd.nc')
# spc = spc.sel(time=slice('2009', '2011'))


#
# from mia_lib.plotlib.miaplot import plot
spc = spc.chunk(dict(encoded_dims=spc.encoded_dims.shape[0]))
spc = spc.isel(time=slice(None,200))
periods = periods.chunk(dict(Period=periods.Period.shape[0]))
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
    # ax.set_xticklabels(['1-4', '4-15', '15-30', '30-90', '> 90'])
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