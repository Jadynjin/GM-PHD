import numpy as np

class KalmanFilter:
    """
    Defines a Kalman filter. 
    
    When fx and hx are not presented, performs KF.
    When either or both are presented, performs EKF.
    When points is presented, performs UKF. F and H will be ignored.
    """
    def __init__(self, x, P, Q, R, F=None, fx=None, H=None, hx=None):
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
    def predict_update(self, dt, z):
        x = self.x
        P = self.P

        F = self.F
        H = self.H

        x_prior = self.fx(x, dt)
        P_prior = F(x, dt) @ P @ F(x, dt).T + self.Q

        S = H(x_prior) @ P_prior @ H(x_prior).T + self.R
        K = P_prior @ H(x_prior).T @ np.linalg.inv(S)
        resident = z - self.hx(x_prior)
        self.x = x_prior + K @ resident
        self.P = (np.eye(self.dim_x) - K @ H(x_prior)) @ P_prior

        