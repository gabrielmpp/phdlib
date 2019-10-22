import numpy as np


# Initial conditions function for diffusion

def squareWave(x, alpha, beta):
    """A square wave as a function of position, x, which is 1 between alpha
    and beta and zero elsewhere. The initialisation is conservative so
    that each phi contains the correct quantity integrated over a region
    a distance dx/2 either side of x"""

    phi = np.zeros_like(x)

    # The grid spacing (assumed uniform)
    dx = (max(x) - min(x)) / (len(x) - 1)

    # Set phi away from the end points (assume zero at the end points)
    for j in range(1, len(x) - 1):
        # edges of the grid box (using west and east notation)
        xw = x[j] - 0.5 * dx
        xe = x[j] + 0.5 * dx

        # integral quantity of phi
        phi[j] = max((min(beta, xe) - max(alpha, xw)) / dx, 0)

    return phi

