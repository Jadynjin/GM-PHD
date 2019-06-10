import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn
from math import atan2, sqrt

from helper import CV_model_2d_func, CV_model_2d_jacobian, Target, Radar
from GaussianMeanShiftRegistration import GaussianMeanShiftRegistration

# Simulation setting
dt = 1
total_time = 40
iters = int(total_time/dt)

# Define targets
target_pos = [30000., 30000.]
target_vel = [40., 40.]
target = Target(target_pos, target_vel)

# Define sensors
sensor1 = Radar(
    pos=[0., 0.], 
    bias=[50., 0.008],
    random=[2., 0.0005])

sensor2 = Radar(
    pos=[50000., 0],
    bias=[50., 0.008],
    random=[2., 0.0005])

# Define filter
x = [30000., 40., 30000., 40.]
Px = [1000, 50, 1e3, 50]

fx = CV_model_2d_func
f = CV_model_2d_jacobian
q = 6.
Qx = np.array(
        [[dt**3/3, dt**2/2, 0, 0],
         [dt**2/2, dt, 0, 0],
         [0, 0, dt**3/3, dt**2/2],
         [0, 0, dt**2/2, dt]])*q

def hx(x):
    x1, y1 = x[0], x[2]
    x2, y2 = x[0] - sensor2.pos[0], x[2] - sensor2.pos[1]
    return np.array([
        sqrt(x1**2 + y1**2),
        atan2(x1, y1),
        sqrt(x2**2 + y2**2),
        atan2(x2, y2)])

def h(x):
    x1, y1 = x[0], x[2]
    x2, y2 = x[0] - sensor2.pos[0], x[2] - sensor2.pos[1]
    mag1 = sqrt(x1**2 + y1**2)
    mag2 = sqrt(x2**2 + y2**2)
    H = np.array([
        [x1/mag1, 0, y1/mag1, 0],
        [y1/mag1**2, 0, -x1/mag1**2, 0],
        [x2/mag2, 0, y2/mag2, 0],
        [y2/mag2**2, 0, -x2/mag2**2, 0]])
    U = np.eye(2)
    return H

R = [2, 5e-4, 2, 5e-4]

gmsr = GaussianMeanShiftRegistration(x, Px, Qx, R, f, h, hx)

filter_result = [gmsr.ekf.x]

# Start simulation
for iter in range(iters):
    # target motion
    target.update(dt)

    # Generate measurement
    # obs_set = []
    obs = sensor1.measure(target)
    obs_set = obs.copy()
    obs = sensor2.measure(target)
    obs_set = np.append(obs_set, obs)
    
    # Filter
    # obs_set = np.array(obs_set)
    gmsr.predict_and_update(dt, obs_set, sensor2_pos=sensor2.pos)
    filter_result.append(gmsr.ekf.x)
    
# Generate data
pos_x = [record[0] for record in target.pos_history]
pos_y = [record[1] for record in target.pos_history]
estimate1 =[record[0] for record in filter_result]  
estimate2 =[record[2] for record in filter_result]  

data_len = iters + 1
time = np.arange(0, total_time+dt/2, dt)
bias1_data = sensor1.bias[0] * np.ones((data_len,))
bias2_data = sensor1.bias[1] * np.ones((data_len,))

# Plot
plt.figure()
# plt.plot(time, bias1_data, label='bias1')
plt.plot(time, pos_x, label='truex')
plt.plot(time, estimate1, label='estimate1')
plt.legend()

plt.figure()
# plt.plot(time, bias2_data, label='bias2')
plt.plot(time, pos_y, label='truey')
plt.plot(time, estimate2, label='estimate2')
plt.legend()
# plt.figure()
# plt.plot(time, target.pos_history, label='bias1')
# plt.plot(time, estimate1, label='estimate1')
# plt.legend()
plt.show()
