# Various function for plotting results and for calculating error measures

import numpy as np

# Import the special package for the erf function
# from scipy import special
# If scipy is unavailable, use math instead
from math import erf


def analyticErf(x, Kt, alpha, beta):
    """The analytic solution of the 1d diffusion equation with diffuions
    coeffienct K at time t assuming top-hat initial conditions which are
    one between alpha and beta and zero elsewhere"""

    phi = np.zeros_like(x)
    for j in range(len(x)):
        phi[j] = 0.5 * erf((x[j] - alpha) / np.sqrt(4 * Kt)) \
                 - 0.5 * erf((x[j] - beta) / np.sqrt(4 * Kt))
    return phi


def L2ErrorNorm(phi, phiExact):
    """Calculates the L2 error norm (RMS error) of phi in comparison to
    phiExact, ignoring the boundaries"""

    # remove one of the end points
    phi = phi[1:-1]
    phiExact = phiExact[1:-1]

    # calculate the error and the error norms
    phiError = phi - phiExact
    L2 = np.sqrt(sum(phiError ** 2) / sum(phiExact ** 2))

    return L2

