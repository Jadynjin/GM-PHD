import numpy as np
from numpy.random import randn
from math import atan2

def CV_model_2d_func(x, dt):
    return np.array([x[0] + x[1]*dt,
                     x[1],
                     x[2] + x[3]*dt,
                     x[3]])

def CV_model_2d_jacobian(x, dt):
    F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])
    G = np.eye(4)
    return F

def bearing_only_2d_measurement(target_pos, pos):
    """
    :param x: vector, target position
    :param pos: vector, self position
    :returns: float, bearing
    """
    x = target_pos - pos
    return atan2(x[0], x[1])

def bearing_only_2d_jacobian_to_cv(target_pos, pos):
    """
    :param x: vector, target position
    :param pos: vector, self position
    :returns: vector, jacobian
    """
    x = target_pos - pos
    mag = x[0]**2 + x[1]**2
    H = np.array([-x[1]/mag, x[0]/mag, 0, 0])
    U = 1
    return (H, U)

class Target:
    """
    Defines a target.

    :param pos: list, position
    :param vel: list, velocity
    """
    def __init__(self, pos, vel):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.pos_history = [self.pos]
    def update(self, dt):
        self.pos += dt * self.vel

class Bearing_only_sensor:
    """
    Defines a bearing_only_sensor.

    :param pos: list, position
    :param bias: float, bias, defaults to 0
    :param random: float, random error, defaults to 0
    """
    def __init__(self, pos, bias=0., random=0.):
        self.pos = pos
        self.bias = bias
        self.random = random
    def measure(self, target):
        """
        :param target: Target, the target being measured
        :returns: float, bearing with bias and random error
        """
        x = target.pos - self.pos
        return atan2(x[0], x[1]) + self.bias + self.random * randn()
