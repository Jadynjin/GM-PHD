# Used for inplement EM method for linear system identification example listed in `An explanation of the Expectation Maximization Algorithm`, the MATLAB code in the paper is referenced.

import numpy as np
from numpy.random import randn

max_iterations = 100
min_ll_decrease = 1e-6 # Min decrease of log-likelihood
guess = 0.1 # Initial guess for the parameter
true = 0.9

# --- REAL SYSTEM ---
N = 500
x = np.zeros(N+1)
y = np.zeros(N)
for t in range(N):
    x[t+1] = true * x[t] + 0.1**0.5*randn(N)
    y[t] = 0.5 * x[t] + 0.1**0.5*randn(N)

# --- EM ALGORITHM ---
LL = []
a = [guess]
for k in range(max_iterations):
    # E step
    g_LL = ks() # TODO
    LL.append(-0.5*g_LL)
    # TODO

    # M step
    guess = psi/phi
    a.append(guess)

    if k>0:
        if LL[k] - LL[k-1] < min_ll_decrease:
            break

    
