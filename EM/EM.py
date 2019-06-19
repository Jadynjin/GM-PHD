# Used for inplement EM method for linear system identification.
# This example is from `An explanation of the Expectation Maximization Algorithm`.
# the MATLAB code in the paper and the relavent toolbox are referenced.

import numpy as np
from numpy.random import randn

from SqrtKalman import sqrt_kalman_smoother

max_iterations = 100
min_ll_decrease = 1e-6 # Min decrease of log-likelihood
guess = np.array([[0.1]]) # Initial guess for the parameter

# --- REAL SYSTEM ---
A = 0.9
N = 500
x = np.zeros(N+1)
y = np.zeros(N)
C = np.array([[0.5]])
Q = 0.1
R = 0.1

X1 = np.array([0])
P1 = np.array([[0]])
for t in range(N):
    x[t+1] = A * x[t]+ Q**0.5*randn()
    y[t] = C * x[t] + R**0.5*randn()

# I cannot implement the old-style random value generator of MATLAB.
# So the real system data is from MATLAB.
from scipy.io import loadmat
x = loadmat('x.mat')['x'][0,:]
y = loadmat('y.mat')['y'].T

# --- EM ALGORITHM ---
LL = []
a = [guess]
for k in range(max_iterations):
    # E step
    # Kalman filter and smoother
    xs, Ps, Ms, g_LL = sqrt_kalman_smoother(y, guess, C, Q, R, X1, P1, N)
    LL.append(-0.5*g_LL)

    # Calculate Q function
    psi, phi = 0, 0
    for t in range(N):
        psi += xs[t] * xs[t+1] + Ms[t,0,0]
        phi += xs[t] ** 2 + Ps[t]**2

    # M step
    guess = psi/phi
    a.append(guess)

    if k>0:
        if LL[k] - LL[k-1] < min_ll_decrease:
            break

print(a)
print(k)
    
