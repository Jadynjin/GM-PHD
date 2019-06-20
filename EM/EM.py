# Used for inplement EM method for linear system identification.
# This example is from `An explanation of the Expectation Maximization Algorithm`.
# the MATLAB code in the paper and the relavent toolbox are referenced.

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.linalg import det

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

X1 = np.array([0])
P1 = np.array([[0]])

aNow = np.r_[0:0.8:0.01, 0.8:1:0.001, 1:1.3:0.01]
len_aNow = len(aNow)
Q_func = []

for k in range(max_iterations):
    # E step
    # Kalman filter and smoother
    xs, Ps, Ms, g_LL = sqrt_kalman_smoother(y, guess, C, Q, R, X1, P1, N, True)
    LL.append(-0.5*g_LL)

    # Compute the Q function
    psi, phi = 0, 0
    alpha, beta = 0, 0
    for t in range(N):
        psi += xs[t] * xs[t+1] + Ms[t,0,0]
        phi += xs[t] ** 2 + Ps[t]**2

        alpha += xs[t+1] * xs[t+1] + Ps[t+1,0,0]**2
        beta += y[t] * xs[t]

    # ???
    Q_k = np.zeros(len_aNow)
    for i in range(len_aNow):
        Q_k[i] = -N * np.log(0.2*np.pi) - 5*sum(y**2) - 5*alpha + 5*beta + 10*psi*aNow[i] - 5*phi*(aNow[i]**2+1/4)
    Q_func.append(Q_k)

    # M step
    guess = psi/phi
    a.append(guess)

    # Check termination condition
    if k>0:
        if LL[k] - LL[k-1] < min_ll_decrease:
            break
    
LL1 = np.zeros(len_aNow)
myLL = np.zeros(len_aNow)
# Compute the log-likelihood function
for i in range(len_aNow):
    A = np.array([[aNow[i]]])
    g_LL, g_Ri, g_xp = sqrt_kalman_smoother(y, A, C, Q, R, X1, P1, N, False)
    LL1[i] = -0.5*g_LL
    # ???
    Ltmp = -(N/2)*np.log(2*np.pi)
    for t in range(1, N):
        cov = g_Ri[t]**2
        Lpart1 = -(1/2)*((y[t] - C @ g_xp[t])**2)/cov
        Lpart2 = -(1/2)*np.log(det(cov))
        Ltmp += Lpart1 + Lpart2
    myLL[i] = Ltmp

# --- PLOT ---
for x in [0,1,2,10]:
    plt.figure()
    plt.plot(aNow, myLL, label='Log-likelihood')
    plt.plot(aNow, Q_func[x], label='Q function')
    iL = np.argmax(myLL)
    mL = myLL[iL]
    iL = aNow[iL]
    plt.plot([iL, iL], [-700, mL + 40])
    iQ = np.argmax(Q_func[x])
    mQ = Q_func[x][iQ]
    iQ = aNow[iQ]
    plt.plot([iQ, iQ], [-700, mQ + 40])
    plt.xlabel('Parameter a')
    plt.ylabel('Log-likelihood and Q funciton')
    plt.legend()

plt.figure()
plt.plot([ia[0][0] for ia in a], label='Parameter estimates')
plt.plot([0.9 for _ in a], label='True value')
plt.xlabel('Iteration')
plt.ylabel('Parameter a')
plt.legend()

plt.figure()
plt.plot(aNow, LL1)
plt.plot(aNow, myLL)
plt.legend()

plt.show()
