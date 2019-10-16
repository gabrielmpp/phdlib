import numpy as np

pa = 1e5
pb = 200.
f = 1e-4
rho = 1.
L = 2.4e6
ymin = 0.0
ymax = 1e6


def pressure(y):
    return pa + pb * np.cos(y*np.pi/L)
def uExact(y):
    return  pb * np.pi/