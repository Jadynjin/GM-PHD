from scipy.io.matlab import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn
from math import atan2

from gmphd import GmphdComponent, Gmphd

def fx(x, dt):
    return np.array([x[0] + x[2]*dt,
                     x[1] + x[3]*dt,
                     x[2],
                     x[3]])

def f(x, dt):
    x = np.reshape(x, (4,))
    F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    G = np.eye(4)
    return (F, G)

def hx(x):
    return np.array([atan2(x[1], x[0]),
                     atan2(x[1] - 500, x[0])])

def h(x):
    x = np.reshape(x, (4,))
    mag = x[0]**2 + x[1]**2
    mag2 = x[0]**2 + (x[1] - 500)**2
    H = np.array([[-x[1]/mag, x[0]/mag, 0, 0],
                  [-(x[1] - 500)/mag2, x[0]/mag2, 0, 0]])
    U = np.eye(2)
    return (H, U)

survive_prob = 0.99
detactive_prob = 1.

birth_gmm = [GmphdComponent(0.05,
                            [200000, 375000, 10, 10],
                            np.diag([100,100,10,10])**2),
             GmphdComponent(0.05,
                            [195000, 375000, 10, 10],
                            np.diag([100, 100, 10, 10])**2)]

Q = np.diag([1e-2, 1e-2, 1e-2, 1e-2])
R = 1e-12*np.eye(2)
clutter = 0

g = Gmphd(birth_gmm, survive_prob, detactive_prob, f, fx, Q, h, hx, R, clutter)

dt = 1
iters = 1438

siml_res_BM = loadmat('siml_res_BM')['siml_res_BM']
BM1_pos_x = siml_res_BM[:, 4]
BM1_pos_y = siml_res_BM[:, 5]
BM1_pos_z = siml_res_BM[:, 6]
BM1_vel_y = siml_res_BM[:, 2]

siml_res_BM2 = loadmat('siml_res_BM2')['siml_res_BM2']
BM2_pos_x = siml_res_BM2[:, 4]
BM2_pos_y = siml_res_BM2[:, 5]
BM2_pos_z = siml_res_BM2[:, 6]


results = []

for iter in range(iters):
    target1 = [BM1_pos_x[iter], BM1_pos_y[iter], 0, 0]
    target2 = [BM2_pos_x[iter], BM2_pos_y[iter], 0, 0]

    obs_set = []
    if rand() < detactive_prob:
        obs = hx(target1) + 1e-6 * randn(2)
        obs_set.append(obs)
    if rand() < detactive_prob:
        obs = hx(target2) + 1e-6*randn(2)
        obs_set.append(obs)
    
    g.update(dt, obs_set)
    g.prune(truncthresh=1e-6, mergethresh=4)

    results.append(g.extractstates())
    
estimate1_x =[record[0][0] for record in results]  
estimate2_x =[record[1][0] for record in results]  
estimate1_y =[record[0][1] for record in results]  
estimate2_y =[record[1][1] for record in results]  
n_estimate = [len(record) for record in results]

time = np.arange(len(BM1_pos_x))

target1_x, target1_y = BM1_pos_x, BM1_pos_y
target2_x, target2_y = BM2_pos_x, BM2_pos_y

plt.figure()
plt.plot(target1_x, target1_y, target2_x, target2_y)

plt.figure()
plt.plot(time, target1_x, label='target1 x true')
plt.plot(time, target2_x, label='target2 x true')
plt.plot(time, estimate1_x, label='estimate1 x')
plt.plot(time, estimate2_x, label='estimate2 x')
plt.legend()

plt.figure()
plt.plot(time, target1_y, label='target1 y true')
plt.plot(time, target2_y, label='target2 y true')
plt.plot(time, estimate1_y, label='estimate1 y')
plt.plot(time, estimate2_y, label='estimate2 y')
plt.legend()

plt.figure()
plt.plot(time, target1_x-estimate1_x, label='error 1 x')
plt.plot(time, target1_y-estimate1_y, label='error 1 y')
plt.plot(time, target2_x-estimate2_x, label='error 2 x')
plt.plot(time, target2_y-estimate2_y, label='error 2 y')
plt.legend()

plt.figure()
plt.plot(time, n_estimate, label='n estimate')
plt.legend()
# plt.figure()
# plt.plot(BM1_pos_x[:-100], BM1_pos_z[:-100], label='BM1')
# plt.plot(BM2_pos_x[:-100], BM2_pos_z[:-100], label='BM2')
# plt.legend()

plt.show()