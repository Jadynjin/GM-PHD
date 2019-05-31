import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn
from math import atan2

from helper import CV_model_2d_func, CV_model_2d_jacobian, Target, Bearing_only_sensor
from KalmanFilter import KalmanFilter
from filterpy.kalman import ExtendedKalmanFilter

np.random.seed(0)
# Simulation setting
dt = 0.01
total_time = 20
iters = int(total_time/dt)

# Define targets
target_pos = [0., 200000.]
target_vel = [5., -7200.]
target = Target(target_pos, target_vel)

# Define sensors
sensor1_pos = [0., 0.]
sensor1_bias = 10/60/57.3
sensor1 = Bearing_only_sensor(
        pos=sensor1_pos,
        bias=sensor1_bias,
        random=1e-5)

# NOTE: This relative position may affect filter performance
sensor2_pos = [500., 0.]
sensor2 = Bearing_only_sensor(
        pos = sensor2_pos,
        bias=-15/60/57.3,
        random=1e-5)


# Define filter
x = [0., 0., 199000., -6000., 0., 0.]
# P = np.diag([10, 10, 1e3, 1200, 1e-2, 1e-2])**2
P = [10, 10, 1e3, 1200, 1e-2, 1e-2]

def f(x, dt):
    return np.array([
        [1., dt, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, dt, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]])

q = 0.00001
Qx = np.array(
        [[dt**3/3, dt**2/2, 0, 0],
         [dt**2/2, dt, 0, 0],
         [0, 0, dt**3/3, dt**2/2],
         [0, 0, dt**2/2, dt]])*q
Qb = np.diag([1e-6, 1e-6])**2
Q = np.array([
    [Qx[0][0], Qx[0][1], Qx[0][2], Qx[0][3], 0, 0],
    [Qx[1][0], Qx[1][1], Qx[1][2], Qx[1][3], 0, 0],
    [Qx[2][0], Qx[2][1], Qx[2][2], Qx[2][3], 0, 0],
    [Qx[3][0], Qx[3][1], Qx[3][2], Qx[3][3], 0, 0],
    [0, 0, 0, 0, Qb[0][0], Qb[0][1]],
    [0, 0, 0, 0, Qb[1][0], Qb[1][1]]])

def hx(x):
    return np.array([atan2(x[0], x[2]) + x[4],
                     atan2(x[0] - 500, x[2]) + x[5]])
                   

def h(x):
    mag = x[2]**2 + x[0]**2
    mag2 = x[2]**2 + (x[0] - 500)**2
    H = np.array([[x[2]/mag, 0, -x[0]/mag, 0, 1, 0],
                  [x[2]/mag2, 0, -(x[0]-500)/mag2, 0, 0, 1]])
    U = np.eye(2)
    return H

# R = np.diag([1e-5, 1e-5])**2
R = [1e-5, 1e-5]

kf = KalmanFilter(x, P, Q, R, F=f, H=h, hx=hx) 
# kf = ExtendedKalmanFilter(6, 2)
# kf.x = x
# kf.P = P
# kf.Q = Q
# kf.R *= 1e-5
# kf.F = f(x, dt)

filter_result = [kf.x]

# Start simulation
for iter in range(iters):
    # target motion
    target.update(dt)

    # Generate measurement
    obs_set = []
    obs = sensor1.measure(target)
    obs_set.append(obs)
    obs = sensor2.measure(target)
    obs_set.append(obs)
    
    # Filter
    obs_set = np.array(obs_set)
    kf.predict_update(dt, obs_set)
#     kf.predict()
#     kf.update(obs_set, h, hx)
    filter_result.append(kf.x)
    
# Generate data
estimate1 =[record[4] for record in filter_result]  
estimate2 =[record[5] for record in filter_result]  

data_len = iters + 1
time = np.arange(0, total_time+dt/2, dt)
bias1_data = sensor1.bias * np.ones((data_len,))
bias2_data = sensor2.bias * np.ones((data_len,))

# Plot
plt.figure()
plt.plot(time, bias1_data, label='bias1')
plt.plot(time, estimate1, label='estimate1')
plt.legend()

plt.figure()
plt.plot(time, bias2_data, label='bias2')
plt.plot(time, estimate2, label='estimate2')
plt.legend()

plt.show()
