from scipy.io.matlab import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn
from math import atan2, sqrt

from gmphd import GmphdComponent, Gmphd

def fx(x, dt):
    return np.array([x[0] + x[3]*dt,
                     x[1] + x[4]*dt,
                     x[2] + x[5]*dt,
                     x[3],
                     x[4],
                     x[5]])

def f(x, dt):
    F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    G = np.eye(6)
    return (F, G)

def hx(x):
    r1 = x[0:3]
    r2 = r1 - np.array([0, 500, 500])

    mag1 = sqrt(r1[0]**2 + r1[2]**2)
    mag2 = sqrt(r2[0]**2 + r2[2]**2)

    return np.array([atan2(r1[1], mag1),
                     -atan2(r1[2], r1[0]),
                     atan2(r2[1], mag2),
                     -atan2(r2[2], r2[0])])

def h(x):
    dP1 = x[0:3]
    dP2 = dP1 - np.array([0, 500, 500])

    mag1 = sqrt(dP1[0]**2 + dP1[1]**2 + dP1[2]**2)
    mag1xz = sqrt(dP1[0]**2 + dP1[2]**2)
    mag2 = sqrt(dP2[0]**2 + dP2[1]**2 + dP2[2]**2)
    mag2xz = sqrt(dP2[0]**2 + dP2[2]**2)
    H = np.array([
                  [-dP1[0]*dP1[1]/(mag1**2*mag1xz), mag1xz/mag1**2, -dP1[2]*dP1[1]/(mag1**2*mag1xz), 0, 0, 0],
                  [dP1[2]/mag1xz**2, 0, -dP1[0]/mag1xz**2, 0, 0, 0],
                  [-dP2[0]*dP2[1]/(mag2**2*mag2xz), mag2xz/mag2**2, -dP2[2]*dP2[1]/(mag2**2*mag2xz), 0, 0, 0],
                  [dP2[2]/mag2xz**2, 0, -dP2[0]/mag2xz**2, 0, 0, 0]])
    U = np.eye(4)
    return (H, U)

survive_prob = 0.99
detactive_prob = 1.

birth_gmm = [GmphdComponent(0.05,
                            [200000, 375000, 0, 500, 500, 500],
                            np.diag([100,100,100,500,500,500])**2),
             GmphdComponent(0.05,
                            [195000, 375000, 0, 500, 500, 500],
                            np.diag([100, 100, 100, 500, 500, 500])**2)]

Q = np.diag([10, 10, 0.0001, 10, 10, 0.0001])**2
measure_std_error = 1e-6
R = 1e-12*np.eye(4)
clutter = 0

g = Gmphd(birth_gmm, survive_prob, detactive_prob, f, fx, Q, h, hx, R, clutter)

dt = 0.02
iters = 1438

siml_res_BM = loadmat('siml_res_BM')['siml_res_BM']
BM1_pos_x = siml_res_BM[:, 4]
BM1_pos_y = siml_res_BM[:, 5]
BM1_pos_z = siml_res_BM[:, 6]
BM1_vel_x = siml_res_BM[:, 1]
BM1_vel_y = siml_res_BM[:, 2]
BM1_vel_z = siml_res_BM[:, 3]

siml_res_BM2 = loadmat('siml_res_BM2')['siml_res_BM2']
BM2_pos_x = siml_res_BM2[:, 4]
BM2_pos_y = siml_res_BM2[:, 5]
BM2_pos_z = siml_res_BM2[:, 6]
BM2_vel_x = siml_res_BM2[:, 1]
BM2_vel_y = siml_res_BM2[:, 2]
BM2_vel_z = siml_res_BM2[:, 3]


results = []

for iter in range(iters):
    target1 = [BM1_pos_x[iter], BM1_pos_y[iter], 0, 0]
    target2 = [BM2_pos_x[iter], BM2_pos_y[iter], 0, 0]

    obs_set = []
    if rand() < detactive_prob:
        obs = hx(target1) + measure_std_error * randn(4)
        obs_set.append(obs)
    if rand() < detactive_prob:
        obs = hx(target2) + measure_std_error*randn(4)
        obs_set.append(obs)
    
    g.update(dt, obs_set)
    g.prune(truncthresh=1e-6, mergethresh=4)

    results.append(g.extractstates())
    
estimate1_x =[record[0][0] for record in results]  
estimate2_x =[record[1][0] for record in results]  
estimate1_y =[record[0][1] for record in results]  
estimate2_y =[record[1][1] for record in results]  
estimate1_z =[record[0][2] for record in results]  
estimate2_z =[record[1][2] for record in results]  

estimate1_vx =[record[0][3] for record in results]  
estimate2_vx =[record[1][3] for record in results]  
estimate1_vy =[record[0][4] for record in results]  
estimate2_vy =[record[1][4] for record in results]  
estimate1_vz =[record[0][5] for record in results]  
estimate2_vz =[record[1][5] for record in results]  

n_estimate = [len(record) for record in results]

time = np.arange(len(BM1_pos_x))

target1_x, target1_y, target1_z = BM1_pos_x, BM1_pos_y, BM1_pos_z
target1_vx, target1_vy, target1_vz = BM1_vel_x, BM1_vel_y, BM1_vel_z
target2_x, target2_y, target2_z = BM2_pos_x, BM2_pos_y, BM2_pos_z
target2_vx, target2_vy, target2_vz = BM2_vel_x, BM2_vel_y, BM2_vel_z

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
plt.plot(time, target1_z, label='target1 z true')
plt.plot(time, target2_z, label='target2 z true')
plt.plot(time, estimate1_z, label='estimate1 z')
plt.plot(time, estimate2_z, label='estimate2 z')
plt.legend()

plt.figure()
plt.plot(time, target1_vx, label='target1 vx true')
plt.plot(time, target2_vx, label='target2 vx true')
plt.plot(time, estimate1_vx, label='estimate1 vx')
plt.plot(time, estimate2_vx, label='estimate2 vx')
plt.legend()

plt.figure()
plt.plot(time, target1_vy, label='target1 vy true')
plt.plot(time, target2_vy, label='target2 vy true')
plt.plot(time, estimate1_vy, label='estimate1 vy')
plt.plot(time, estimate2_vy, label='estimate2 vy')
plt.legend()

plt.figure()
plt.plot(time, target1_vz, label='target1 vz true')
plt.plot(time, target2_vz, label='target2 vz true')
plt.plot(time, estimate1_vz, label='estimate1 vz')
plt.plot(time, estimate2_vz, label='estimate2 vz')
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

plt.show()