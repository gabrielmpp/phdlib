import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dt = 3600
timestamps = pd.date_range("2019-01-01T00:00:00", "2021-12-31T00:00:00", freq=str(dt) + 's')
A = np.zeros([len(timestamps)])
P = np.zeros([len(timestamps)])
Q = np.zeros([len(timestamps)])
dphi = np.zeros([len(timestamps)])

v_synoptic = np.zeros([len(timestamps)])

A[0] = 5
A_amazon = 10
V_amazon = np.zeros([len(timestamps)])
dx = 2e6
E = 0

activation = 'linear'

def V_Monsoon(t):
    """

    :param t: timestep index
    :return: Mass flux from the Amazon
    """
    amplitude = 7
    half_year = 0.5 * 365 * 86400
    return 10 + amplitude * np.cos(t*dt*np.pi/half_year)


def V_synoptic(t, dphi):
    """

    :param t: timestep index
    :return: Mass flux from the Amazon
    """
    amplitude = dphi
    half_period = 0.5 * 7 * 86400
    return 5 + amplitude * np.cos(t*dt*np.pi/half_period)



def sigmoid(x):
    return 1 / (1 + np.exp(-0.01 * x))


def precip_param(A, Q, threshold=4.5, alpha=1):
    if A > threshold:
        return Q
    else:
        return 0


def linear(x): return x


funcs = dict(sigmoid=sigmoid,
             ReLU=precip_param,
             linear=linear)
f = funcs[activation]
coupling = 1000
for k, date in enumerate(timestamps[1:]):
    rossby_period = 0.5 * 7 * 86400

    #P[k+1] = np.random.gamma([1])*f(A[k]) / 86400
    P[k] = 0.5 * f(A[k]) / 86400
    V_amazon[k] = V_Monsoon(k)
    V = V_amazon[k] + v_synoptic[k]/2
    Q[k] = - (V_SP * A[k] - V * A_amazon) / dx
    print(coupling * P[k])
    #v_synoptic[k+1] =  (coupling * P[k])+v_synoptic[k] + (0.1 ) * np.cos(k * dt * np.pi / rossby_period)
    v_synoptic[k+1] = v_synoptic[k] + (coupling * P[k] + 0.1) * np.sin(k * dt * np.pi / rossby_period)
    A[k + 1] = max(A[k] + dt * (E - P[k] + Q[k]), 0)

import xarray as xr

A_xr = xr.DataArray(A, dims=['time'], coords=dict(time=timestamps))
P_xr = xr.DataArray(P, dims=['time'], coords=dict(time=timestamps))
V_xr = xr.DataArray(V_amazon + v_synoptic, dims=['time'], coords=dict(time=timestamps))
Q_xr = xr.DataArray(Q, dims=['time'], coords=dict(time=timestamps))

A_xr.isel(time=slice(0, None)).plot()
(P_xr * 86400).isel(time=slice(0, None)).plot()
(Q_xr * 86400).isel(time=slice(0, None)).plot()
plt.legend(['A', 'P', 'Q'])
plt.show()

A_xr.sel(time='2019-07').plot()
(P_xr * 86400).sel(time='2019-07').plot()
(Q_xr * 86400).sel(time='2019-07').plot()
plt.legend(['A', 'P', 'Q'])
plt.show()
A_xr.groupby('time.month').mean().plot()
(Q_xr * 86400).groupby('time.month').mean().plot()

(P_xr * 86400).groupby('time.month').mean().plot()
plt.legend(['A', 'Q', 'P'])
plt.show()
plt.plot(A, Q)
plt.show()

timestamps = pd.date_range("2019-01-01T00:00:00", "2019-12-31T00:00:00", freq=str(dt) + 's')
