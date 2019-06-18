import numpy as np
import scipy.linalg.inv

class KalmanFilter:
    """
    Defines a Kalman filter. 
    
    When fx and hx are not presented, performs KF.
    When either or both are presented, performs EKF.
    When points is presented, performs UKF. F and H will be ignored.
    """
    def __init__(self, x, P, Q, R, F=None, fx=None, H=None, hx=None):
        """
        :param x: list
        :param P: list, sqrt of diagnose of P, std of each variable
        :param Q: 2d ndarray
        :param R: list, sqrt of diagnose of R, std of each measurement
        :param F: (x, dt) => 2d ndarray
        :param H: (x) => 2d ndarray
        :param hx: (x, args) => 1d ndarray
        """
        self.dim_x = len(x)
        self.dim_z = len(R)

        self.x = np.array(x)
        self.P = np.diag(P)**2

        self.F = F
        if fx == None:
            self.fx = lambda x, dt: F(x, dt) @ x
        else:
            self.fx = fx
        self.Q = Q

        self.H = H
        if hx == None:
            self.hx = lambda x: H(x) @ x
        else:
            self.hx = hx
        self.R = np.diag(R)**2

        self.x_prior = np.zeros((self.dim_x,))
        self.P_prior = np.zeros((self.dim_x, self.dim_x))

    def predict_update(self, dt, z, h_args=(), hx_args=()):
        self.predict(dt)
        self.update(z, h_args, hx_args)

    def predict(self, dt):
        x = self.x
        P = self.P

        F = self.F

        self.x_prior = self.fx(x, dt)
        self.P_prior = F(x, dt) @ P @ F(x, dt).T + self.Q

    def update(self, z, h_args, hx_args):
        x_prior = self.x_prior
        P_prior = self.P_prior

        H = self.H

        S = H(x_prior) @ P_prior @ H(x_prior).T + self.R
        K = P_prior @ H(x_prior).T @ inv(S)
        residual = z - self.hx(x_prior, *hx_args)
        self.x = x_prior + K @ residual
        I_KH = np.eye(self.dim_x) - K @ H
        self.P = I_KH @ P_prior @ I_KH.T + K @ self.R @ K.T

        