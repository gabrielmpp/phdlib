import numpy as np
import math
import matplotlib.pyplot as plt


def taylor_exp(x):
    """
    Funcion to compute the exponential of x using 6 terms from the Taylor
    expansion.
    :param x: float
    :return: float
    """
    res = 1 + x + 0.5 * x ** 2 + (1 / 6) * x ** 3 + (1 / 24) * x ** 4 + (1 / math.factorial(5)) * x ** 5 + (
            1 / math.factorial(6)) * x ** 6
    return res


def taylorExp(x, n):
    """
    Funcion to compute the exponential of x using n terms from the Taylor
    expansion.

    :param x: float
    :param n: integer, number of terms
    :return: float
    """
    result = 0
    for i in range(n):
        result += (x ** i) / math.factorial(i)
    return result


def gaussianQuadrature(a, b, N, f, precision="double"):
    """
    Apply the Gaussian Quadrature in a function f

    :param a: float, initial point
    :param b: float, final point
    :param N: int, number of intervals
    :param f: function to integrate
    :param precision: str, "single" or "double
    :return: float, result of integration
    """

    I = 0
    dx = (b - a) / N
    if precision == 'double':
        for i in range(N):
            I += f(a + (i + 0.5) * dx)

    elif precision == 'single':
        a = np.float32(a)
        dx = np.float32(dx)
        for i in range(N):
            I += np.float32(f(a + (i + 0.5) * dx))

    return I * dx


if __name__ == '__main__':

    # Evaluating exp(2)
    print(taylor_exp(2))

    # Comparing w/ numpy
    print(taylor_exp(2) - np.exp(2))

    # Testing more terms
    print(taylorExp(2, 10) - np.exp(2))

    # Gaussian quadrature for sin(pi/2)
    double = gaussianQuadrature(0, 0.5 * np.pi, np.sin)
    print(double)

    # Gaussian quadrature single precision
    single = gaussianQuadrature(0, 0.5 * np.pi, np.sin, precision="single")

    print(single - double)

    # Evaluating Integral of exp from 0 to 1 w/ different number of intervals
    results = []
    Ns = np.array([2, 20, 200, 2000])
    for N in Ns:
        results.append(gaussianQuadrature(0, 1, N, np.exp))

    # Computing error and plotting in log scale
    error = np.abs(results - (np.exp(1) - np.exp(0)))
    plt.plot(Ns, error)
    plt.yscale('logit')
    plt.show()
