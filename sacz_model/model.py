import pandas as pd
import numpy as np
import xarray as xr
import numba

config = dict(
    A_0=5,
    dt=3600,  # seconds,
    dx=2e6,
    start_date="2005-01-01T00:00:00",
    end_date="2010-12-31T00:00:00",
    coupling=0.05
)


def check_nodes(P, Q, R):
    """
    Check trajectories based on Chong et al. 1990
    :param P:
    :param Q:
    :param R:
    :return:
    """
    Ra = (1/3) * P * (Q - (1/9) * P**2) - (2/27)*(-3*Q + P ** 2)**(3/2)
    Rb = (1/3) * P * (Q - (1/9) * P**2) + (2/27)*(-3*Q + P ** 2)**(3/2)

    if P > 0 and 0 < Q < (P ** 2) / 3 and 0 < R < Rb:
        return "Stable node/node/node"

    elif P < 0 and 0 < Q < (P ** 2) / 3 and Ra < R < 0:
        return "Unstable node/node/node"
    elif P > 0 and (P ** 2) / 4 < Q < (P ** 2) / 3 and R == Ra:
        return "Unstable node/node/star node"


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
        self.diffusion_wind = 0.5

        self.rossby_period = 0.5 * 4 * 86400
        self.rossby_amplitude = 1e-5

    def analytical(self):
        alpha1 = 0.1
        alpha2 = -self.diffusion_wind
        alpha3 = 0.1 * self.coupling
        alpha4 = 0.5 # thats the precip param and it is a lie for now
        alpha5 = -self.diffusion
        eigenvalues = np.zeros([100, 3])
        for i, drying_factor in enumerate(np.linspace(0, 1, 100)):
            alpha6 = drying_factor * self.A_amazon/self.dx

            A = np.array(
                [
                    [alpha2, alpha1, 0],
                    [0, alpha5, alpha3*alpha4],
                    [alpha6, 0, -alpha4]
                ]
            )
            # Following Chong et al. (1990)
            eigvalues = np.linalg.eig(A)[0]
            # sigma = np.real(eigvalues[np.abs(np.imag(eigvalues)) > 0][0])
            # omega = np.abs(np.imag(eigvalues[np.abs(np.imag(eigvalues)) > 0][0]))
            # b = np.real(eigvalues[np.abs(np.imag(eigvalues)) == 0][0])
            # P = -(2*sigma + b)
            # Q = sigma**2 + omega**2 + 2*sigma*omega
            # R = -b*(sigma**2 + omega**2)
            eigenvalues[i, ] = np.imag(eigvalues)

        eigenvalues = xr.DataArray(eigenvalues, dims=['drying_factor', 'eigenvalue'], coords=dict(
            drying_factor=np.linspace(0, 1, 100), eigenvalue=[0, 1, 2]
        ))
        eigenvalues.plot.line(x='drying_factor')
        plt.ylabel('Imaginary part')
        plt.show()
        print( np.linalg.eig(A)[0])
        eigvalue1 = np.linalg.eig(A)[0][0]
        eigvector1 = np.linalg.eig(A)[1][0]





    @numba.jit
    def integrate(self):

        # P[k+1] = np.random.gamma([1])*f(A[k]) / 86400
        for k in range(len(self.timestamps[1:])):
            # Diagnostic equations
            self.P[k], self.A[k] = self.precip_param(self.A[k])
            self.V_amazon[k] = self.Vmonsoon(k)
            self.v_synoptic[k+1] = self.v_synoptic[k] + 0.1* self.dphi[k] - self.diffusion_wind * self.v_synoptic[k]
            V = self.V_amazon[k] + 100*self.v_synoptic[k]
            self.Q[k] = V * self.A_amazon / self.dx
            # Prognostic equations
            self.dphi[k + 1] = self.dphi[k] + 0.1 * self.dt * (
                    self.coupling * self.P[k] + self.rossby_amplitude * np.sin(
                k * self.dt * np.pi / self.rossby_period)) \
                               - self.diffusion * self.dphi[k]

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
        return amplitude * np.cos(t * self.dt * np.pi / half_year)

    def precip_param(self, A, threshold=12, alpha=1):

        if A > threshold:
            exceeding = (A - threshold)
            return 0.9 * exceeding / self.dt, 0 * threshold
        else:
            return 0, A


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def linear(x): return x


def run(config, coupling, A_amazon):
    sacz = Sacz(config)
    sacz.coupling = coupling
    sacz.A_amazon = A_amazon
    sacz = sacz.integrate()
    return sacz.P, sacz.dphi, sacz.A, sacz.Q


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed
    import itertools
    sacz = Sacz(config)
    sacz.analytical()

    coupling_list = np.arange(0,5,1)  # entre 0 e 80
    A_amazon_list = [7, 10]
    iterlist = list(itertools.product(coupling_list, A_amazon_list))

    P_list = []
    dphi_list = []
    A_list = []
    Q_list = []
    result = Parallel(n_jobs=5, prefer="threads")(
        delayed(run)(config, *x) for x in iterlist)
    for r in result:
        P_list.append(r[0])
        dphi_list.append(r[1])
        A_list.append(r[2])
        Q_list.append(r[3])

    A = xr.concat(A_list, dim=pd.MultiIndex.from_product(
        [coupling_list, A_amazon_list],
        names=['coupling', 'A_amazon'])).unstack()
    dphi = xr.concat(dphi_list, dim=pd.MultiIndex.from_product(
        [coupling_list, A_amazon_list],
        names=['coupling', 'A_amazon'])).unstack()
    P = xr.concat(P_list, dim=pd.MultiIndex.from_product(
        [coupling_list, A_amazon_list],
        names=['coupling', 'A_amazon'])).unstack()
    Q = xr.concat(Q_list, dim=pd.MultiIndex.from_product(
        [coupling_list, A_amazon_list],
        names=['coupling', 'A_amazon'])).unstack()


    (A).sel(time='2008', A_amazon=A_amazon_list[1]).plot.line(x='time')
    plt.show()
    (Q * 86400).sel(time='2008', A_amazon=A_amazon_list[1]).plot.line(x='time')
    plt.show()
    (P * 86400).sel(time='2008', A_amazon=A_amazon_list[1]).resample(time='1D').sum().plot.line(x='time')
    plt.show()
    (dphi).sel(time='2008', A_amazon=A_amazon_list[1]).plot.line(x='time')
    plt.show()
    (P * 86400).sum('time').plot.line(x='coupling')
    plt.show()
    '''
    ds = xr.Dataset(dict(
        A=A,
        Q=Q,
        P=P,
        dphi=dphi
    ))
    ds.to_netcdf('data/ds.nc')

    
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
