import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn
from math import atan2

from helper import CV_model_2d_func, CV_model_2d_jacobian, Target, Bearing_only_sensor
from TwoStageKalmanFilter import OptimalTwoStageKalmanFilter

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

# This relative position can give a better result.
# sensor2_pos = [500., 0.]
# While this cannot.
sensor2_pos = [5000., 0.]
sensor2 = Bearing_only_sensor(
        pos = sensor2_pos,
        bias=-15/60/57.3,
        random=1e-5)


# Define filter
x = [0., 0., 199000., -6000.]
Px = [10, 10, 1e3, 1200]

b = [0., 0.]
Pb = [1e-2, 1e-2]

fx = CV_model_2d_func
f = CV_model_2d_jacobian
q = 0.00001
Qx = np.array(
        [[dt**3/3, dt**2/2, 0, 0],
         [dt**2/2, dt, 0, 0],
         [0, 0, dt**3/3, dt**2/2],
         [0, 0, dt**2/2, dt]])*q
Qb = np.diag([1e-6, 1e-6])**2

def hx(x, dt, sensor2_pos):
    return np.array([atan2(x[0], x[2]),
                     atan2(x[0] - sensor2_pos[0], x[2] - sensor2_pos[1])])

def h(x, dt, sensor2_pos):
    mag = x[2]**2 + x[0]**2
    mag2 = (x[2] - sensor2_pos[1])**2 + (x[0] - sensor2_pos[0])**2
    H = np.array([[x[2]/mag, 0, -x[0]/mag, 0],
                  [(x[2] - sensor2_pos[1])/mag2, 0, -(x[0] - sensor2_pos[0])/mag2, 0]])
    U = np.eye(2)
    return H

R = [1e-5, 1e-5]

otskf = OptimalTwoStageKalmanFilter(x, Px, b, Pb, Qx, Qb, R, F=f, H=h, Hx=hx) 
filter_result = [otskf.b]

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
    otskf.predict_and_update(dt, obs_set, sensor2_pos=sensor2.pos)
    filter_result.append(otskf.b)
    
# Generate data
estimate1 =[record[0] for record in filter_result]  
estimate2 =[record[1] for record in filter_result]  

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
