import pandas as pd
import numpy as np
import xarray as xr
import numba

config = dict(
    A_0=5,
    dt=3600,  # seconds,
    dx=2e6,
    start_date="2005-01-01T00:00:00",
    end_date="2015-12-31T00:00:00",
    coupling=0.1
)


class Sacz:
    def __init__(self, config):
        self.dt = config['dt']

        self.timestamps = pd.date_range(config['start_date'], config['end_date'], freq=str(self.dt) + 's')
        self.A = np.zeros([len(self.timestamps)])
        self.P = np.zeros([len(self.timestamps)])
        self.Q = np.zeros([len(self.timestamps)])
        self.dphi = np.zeros([len(self.timestamps)])

        self.V_amazon = np.zeros([len(self.timestamps)])
        self.v_synoptic = np.zeros([len(self.timestamps)])
        self.A_amazon = 10
        self.E = 0
        self.A[0] = config['A_0']
        self.dx = config['dx']
        self.coupling = config['coupling']
        self.diffusion = 1e-2
        self.rossby_period = 0.5 * 7 * 86400

    @numba.jit(parallel=True)
    def integrate(self):

        # P[k+1] = np.random.gamma([1])*f(A[k]) / 86400
        for k in range(len(self.timestamps[1:])):
            # Diagnostic equations
            self.P[k], self.A[k] = self.precip_param(self.A[k])
            self.V_amazon[k] = self.Vmonsoon(k)
            self.v_synoptic[k] = 40*self.dphi[k]
            V = self.V_amazon[k] + self.v_synoptic[k] / 2
            self.Q[k] = V * self.A_amazon / self.dx
            # v_synoptic[k+1] =  (coupling * P[k])+v_synoptic[k] + (0.1 ) * np.cos(k * dt * np.pi / rossby_period)
            # Prognostic equations
            self.dphi[k + 1] = self.dphi[k] + 0.01*self.dt * (
                    self.coupling*self.P[k] + 0.5e-4*np.sin(k * self.dt * np.pi / self.rossby_period)) \
            -self.diffusion * self.dphi[k]

            self.A[k + 1] = max(self.A[k] + self.dt * (self.E - self.P[k] + self.Q[k]), 0)

        self.A = xr.DataArray(self.A, dims=['time'], coords=dict(time=self.timestamps))
        self.P = xr.DataArray(self.P, dims=['time'], coords=dict(time=self.timestamps))
        self.V = xr.DataArray(self.V_amazon + self.v_synoptic, dims=['time'], coords=dict(time=self.timestamps))
        self.Q = xr.DataArray(self.Q, dims=['time'], coords=dict(time=self.timestamps))
        self.dphi = xr.DataArray(self.dphi, dims=['time'], coords=dict(time=self.timestamps))

        return self

    def forcing(self, P, k):
        dphi = 10 * np.sin(k * self.dt * np.pi / self.rossby_period)
        if P > 0:
            return dphi + self.coupling * sigmoid(P)
        else:
            return dphi

    def Vmonsoon(self, t):
        """
        Function to calculate the monsoon mass flux from Amazon
        :param t: timestep index
        :return: Mass flux from the Amazon
        """
        amplitude = 20
        half_year = 0.5 * 365 * 86400
        return amplitude * np.cos(t*self.dt*np.pi/half_year)

    def precip_param(self, A, threshold=12, alpha=1):

        if A > threshold:
            exceeding = (A-threshold)
            return 0.9*exceeding/self.dt, 0*threshold
        else:
            return 0, A


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def linear(x): return x


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import time
    sacz = Sacz(config)
    sacz = sacz.integrate()
    coupling_list = [0, 0.1, 1, 10, 20, 40, 80]
    P_list = []
    dphi_list = []
    A_list = []
    for coupling in coupling_list:
        print("Running model with coupling = {}".format(str(coupling)))
        config['coupling'] = coupling
        sacz = Sacz(config)
        start = time.time()
        sacz = sacz.integrate()
        end = time.time()
        print("Integration time is = {}".format(str(end - start)))

        P_list.append(sacz.P)
        dphi_list.append(sacz.dphi)
        A_list.append(sacz.A)
    A = xr.concat(A_list, dim=pd.Index(coupling_list, name='coupling'))
    dphi = xr.concat(dphi_list, dim=pd.Index(coupling_list, name='coupling'))
    P = xr.concat(P_list, dim=pd.Index(coupling_list, name='coupling'))
    (P*86400).sel(time='2010').plot.line(x='time')
    plt.show()
    (P*86400).mean('time').plot()
    plt.show()
    dphi.sel(time='2010').plot.line(x='time')
    plt.show()




    '''
    #sacz.V.sel(time='2019').plot()
    (sacz.Q*86400).sel(time='2010').resample(time='1D').mean().plot()
    (sacz.P*86400).sel(time='2010').resample(time='1D').sum().plot()
    (sacz.Q*86400).sel(time='2010').resample(time='5D').mean().plot()
    (sacz.P*86400).sel(time='2010').resample(time='5D').sum().plot()
    #sacz.A.sel(time='2019').plot()
    plt.legend(['Q', 'P'])
    plt.show()

    plt.plot(sacz.A.resample(time='5D').mean().values,
             sacz.Q.resample(time='5D').mean().values)
    plt.show()
    sacz_dez = sacz.P.sel(time='2010-12')
    plt.hist(sacz_dez.where(sacz_dez * 86400 > 1, drop=True).time.diff('time').values.tolist())
    plt.show()

    sacz.dphi.plot()
    plt.show()
    '''
