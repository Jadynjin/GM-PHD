# Used for inplement EM method for linear system identification example listed in `An explanation of the Expectation Maximization Algorithm`, the MATLAB code in the paper is referenced.

import numpy as np
from numpy.random import randn

from SqrtKalman import sqrt_kalman_smoother

max_iterations = 100
min_ll_decrease = 1e-6 # Min decrease of log-likelihood
guess = 0.1 # Initial guess for the parameter

# --- REAL SYSTEM ---
A = 0.9
N = 500
x = np.zeros(N+1)
y = np.zeros(N)
C = 0.5
Q = 0.1
R = 0.1

X1 = 0
P1 = 0
for t in range(N):
    x[t+1] = A * x[t]+ Q**0.5*randn()
    y[t] = C * x[t] + R**0.5*randn()

from scipy.io import loadmat
x = loadmat('EMLinSysId/x.mat')['x'][0,:]
y = loadmat('EMLinSysId/y.mat')['y'][0,:]

# --- EM ALGORITHM ---
LL = []
a = [guess]
for k in range(max_iterations):
    # E step
    g_LL = sqrt_kalman_smoother(y, guess, C, Q, R, X1, P1, N) # TODO
    LL.append(-0.5*g_LL)
    # TODO

    # M step
    guess = psi/phi
    a.append(guess)

    if k>0:
        if LL[k] - LL[k-1] < min_ll_decrease:
            break

    
