import numpy as np
from numpy import pi
from numpy.random import rand, randn
from math import sin, cos, atan2
import matplotlib.pyplot as mlt

from gmphd import GmphdComponent, Gmphd

def fx(x, dt):
    omega = x[4]
    tol = 1e-10
    sin_omega_T = sin(omega*dt)
    cos_omega_T = cos(omega*dt)
    if abs(omega) > tol:
        a = sin_omega_T/omega
        b = (1 - cos_omega_T)/omega
    else:
        a = 1
        b = 0
    return np.array([x[0] + a * x[1],
                           cos_omega_T*x[1] - sin_omega_T*x[3],
                           b*x[1] + x[2] + a*x[3],
                           sin_omega_T*x[1] + cos_omega_T*x[3],
                           x[4]])

def f(x, dt):
    "Jacobian of fx"
    tol = 1e-6
    x = np.reshape(x, (5,))
    omega = x[4]
    sin_omega_T = sin(omega*dt)
    cos_omega_T = cos(omega*dt)
    if abs(omega) > tol:
        a = sin_omega_T/omega
        b = (1 - cos_omega_T)/omega
        c = (omega*dt*cos_omega_T - sin_omega_T)/omega**2
        d = (omega*dt*sin_omega_T - 1 + cos_omega_T)/omega**2
    else:
        a = dt
        b = 0
        c = 0
        d = dt**2/2
    dA = np.array([[0, c, 0, -d],
                   [0, -dt*sin_omega_T, 0, -dt*cos_omega_T],
                   [0, d, 0, c],
                   [0, dt*cos_omega_T, 0, -dt*sin_omega_T]])
    dAmu = dA @ x[0:4]
    F = np.array([[1, a, 0, b, dAmu[0]],
                  [0, cos_omega_T, 0, -sin_omega_T, dAmu[1]],
                  [0, b, 1, a, dAmu[2]],
                  [0, sin_omega_T, 0, cos_omega_T, dAmu[3]],
                  [0, 0, 0, 0, 1]])
    sigma_vel, sigma_turn = 5, pi/180
    G = np.array([[sigma_vel*dt**2/2, 0, 0],
                  [sigma_vel*dt, 0, 0],
                  [0, sigma_vel*dt**2/2, 0],
                  [0, sigma_vel*dt, 0],
                  [0, 0, dt*sigma_turn]])
    return (F, G)
    
def hx(x):
    return np.array([atan2(x[0], x[2]),
                     atan2(x[0] - 500, x[2])])

def h(x):
    "Jacobian of hx"
    p = x[0:4:2]
    p = np.reshape(p, (2,))
    mag = p[0]**2 + p[1]**2
    mag2 = (p[0]-500)**2 + p[1]**2
    H = np.array([[p[1]/mag, 0, -p[0]/mag, 0, 0],
                  [p[1]/mag2, 0, -(p[0]-500)/mag2, 0, 0]])
    U = np.eye(2)
    return (H, U)

class Target():
    def __init__(self, state):
        self.state = state

    def state_update(self, dt):
        self.state = fx(self.state, dt)

    def observe(self):
        return hx(self.state)


survive_prob = 0.99 # P_S
detactive_prob = 1. # P_D

birth_gmm = [GmphdComponent(0.02,
                        [-1500, 0, 250, 0, 0],
                        np.diag([50, 50, 50, 50, 6*np.pi/180])**2),
             GmphdComponent(0.02,
                        [-250, 0, 1000, 0, 0],
                        np.diag([50, 50, 50, 50, 6*np.pi/180])**2),
             GmphdComponent(0.03,
                        [250, 0, 750, 0, 0],
                        np.diag([50, 50, 50, 50, 6*np.pi/180])**2),
             GmphdComponent(0.03,
                        [1000, 0, 1500, 0, 0],
                        np.diag([50, 50, 50, 50, 6*np.pi/180])**2)]


Q = np.eye(3)
R = 1e-12*np.diag([0.2*np.pi/180, 0.2*np.pi/180])**2
clutter = 1.0/(pi/2+pi/2)/2000*0

g = Gmphd(birth_gmm,survive_prob, detactive_prob, f, fx, Q, h, hx, R, clutter)

dt = 1
results = []
iters = 100

wturn = 2*pi/180
target1 = Target(np.array([1000+3.8676, -10, 1500-11.7457, -10, wturn/8]))
target2 = Target(np.array([-250-5.8857, 20, 1000+11.4102, 3, -wturn/3]))

true_items = [[target1.state, target2.state]]
for iter in range(iters):
    # generate true states
    target1.state_update(dt)
    target2.state_update(dt)
    true_items.append([target1.state, target2.state])

    # generate observations
    obs_set = []
    if rand() < detactive_prob:
        obs = target1.observe() + 1e-6*0.2*pi/180*randn(2)
        obs_set.append(obs)
    if rand() < detactive_prob:
        obs = target2.observe() + 1e-6*0.2*pi/180*randn(2)
        obs_set.append(obs)
    
    # PHD filter
    g.update(dt, obs_set)
    g.prune(truncthresh=1e-6, mergethresh=4)


time = np.arange(iters+1)
target1_state1 = [record[0][0] for record in true_items] 
target1_state2 = [record[0][2] for record in true_items] 
target2_state1 = [record[1][0] for record in true_items] 
target2_state2 = [record[1][2] for record in true_items] 
mlt.figure()
mlt.plot(target1_state1, target1_state2, target2_state1, target2_state2)
mlt.show()