import numpy as np
import math


def taylor_exp(x):
    return 1 + x + 0.5*x**2 + (1/6)*x**3 + (1/24)*x**4 + (1/math.factorial(5))*x**5 + (1/math.factorial(6))*x**6


def taylorExp(x, n):
    result=0
    for i in range(n):
        result+=(x**i)/math.factorial(i)
    return result


def gaussianQuadrature(a, b, N, f, precision="double"):

    I = 0
    dx = (b - a)/N
    if precision=='double':
        for i in range(N):
            I += f(a + (i+0.5)*dx)

    elif precision == 'single':
        a = np.float32(a)
        dx = np.float32(dx)
        for i in range(N):
            I += np.float32(f(a + (i+0.5)*dx))

    return I*dx

#single vs double
print(gaussianQuadrature(0,np.pi*0.5,200,np.sin, precision='single')-gaussianQuadrature(0,np.pi*0.5,200,np.sin, precision='double'))

#calc error
results = []
Ns = [2,20,200, 2000]
for N in  Ns:
    results.append(gaussianQuadrature(0,1,N, np.exp))
error = np.abs(results - (np.exp(1)-np.exp(0)))
import matplotlib.pyplot as plt
plt.plot(np.sqrt(error))
plt.show()