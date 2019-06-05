import numpy as np
from copy import deepcopy

class OptimalTwoStageKalmanFilter:
    """
    Defines an optimal two stage Kalman filter. Also deals with nonlinear case. When Fx or Hx or both is presented, performs EKF version. When points presented, performs UKF version.

    :param x1: list, initial state estimation (nx1)
    :param P1: list, diagnose of covariance matrix of state vector
    :param b: list, initial bias estimation (mx1)
    :param P2: list, diagnose of covariance matrix of bias vector
    :param Q_x: nxn matrix, covariance of state process noise
    :param Q_b: mxm matrix, covariance of bias process noise
    :param R: list, diagnose of covariance matrix of measurement noise (mxm)
    :param F: float, float -> nxn matrix, from state, dt returns state transmition matrix
    :param Fx:
    :param H: mxn matrix, measurement matrix
    :param Hx:
    :param points:
    :param B: nxm matrix, bias to state matrix, default to zeros((m, m))
    :param C: mxm matrix, bias transmition matrix, default to eye(m)
    :param D: mxm matrix, bias to measurement matrix, default to eye(m)
    
    ATRRIBUTES
    dim_x:
    dim_z:
    x1:
    P1:
    b:
    P2:
    x:
    P:
    V:
    F:
    Fx:
    Q_x:
    Q_b:
    H:
    Hx:
    R:
    B:
    C:
    D:
    points:
    """
    def __init__(self, x1, P1, b, P2, Q_x, Q_b, R, F=None, Fx=None, H=None, Hx=None, B=None, C=None, D=None, points=None):
        """
        :param x1: list
        """
        dim_x = len(x1)
        dim_z = len(b)

        self.x1 = np.array(x1)
        self.P1 = np.diag(P1)**2
        self.b = np.array(b)
        self.P2 = np.diag(P2)**2
        # Assumes at initial time, state and bias are independent
        self.V = np.zeros((dim_x, dim_z))
        self.x = deepcopy(self.x1)
        self.P = deepcopy(self.P1)

        self.F = F
        if Fx == None:
            self.Fx = lambda x, dt: F(x, dt) @ x
        else:
            self.Fx = Fx
        self.Qx = Q_x
        self.Qb = Q_b

        self.H = H
        if Hx == None:
            self.Hx = lambda x, dt: H(x, dt) @ x
        else:
            self.Hx = Hx
        self.R = np.diag(R)**2

        if B == None:
            self.B = np.zeros((dim_x, dim_z))
        if C == None:
            self.C = np.eye(dim_z)
        if D == None:
            self.D = np.eye(dim_z)

        self.dim_x = dim_x
        self.dim_z = dim_z
    def predict_and_update(self, dt, y, **args):
        """
        dt: float, step time
        y: m vector, measurement
        """
        Qx = self.Qx
        Qb = self.Qb
        R = self.R
        F = self.F
        Fx = self.Fx
        H = self.H
        Hx = self.Hx
        B = self.B
        C = self.C
        D = self.D

        x = self.x
        P = self.P
        x1 = self.x1
        P1 = self.P1
        b = self.b
        P2 = self.P2
        V = self.V
        ### --------------- PREDICT ----------------
        # bias filter
        b_prior = C @ b
        P2_prior = C @ P2 @ C.T + Qb
        # coupling
        Ubar = F(x, dt) @ V + B 
        U = Ubar @ P2 @ C.T @ np.linalg.inv(P2_prior)
        # bias-free filter
        x1_prior = Fx(x, dt) + B @ b - U @ b_prior
        P1_prior = F(x, dt) @ P1 @ F(x, dt).T + Qx + Ubar @ P2 @ Ubar.T - U @ P2_prior @ U.T
        ### ---------------- UPDATE -----------------
        x_prior = x1_prior + U @ b_prior
        # bias filter
        S = H(x_prior, dt, **args) @ U + D
        K2 = P2_prior @ S.T @ np.linalg.inv(S @ P2_prior @ S.T + R + H(x_prior, dt, **args) @ P1_prior @ H(x_prior, dt, **args).T) 
        resi2 = (y - Hx(x_prior, dt, **args) - D @ b_prior)
        b = b_prior + K2 @ resi2
        P2 = (np.eye(self.dim_z) - K2 @ S) @ P2_prior
        P2 = (P2 + P2.T)/2
        # bias-free filter
        K1 = P1_prior @ H(x_prior, dt, **args).T @ np.linalg.inv(H(x_prior, dt, **args) @ P1_prior @ H(x_prior, dt, **args).T + R)
        x1 = x1_prior + K1 @ (resi2 + S @ b_prior)
        P1 = (np.eye(self.dim_x) - K1 @ H(x_prior, dt, **args)) @ P1_prior
        P1 = (P1 + P1.T)/2
        # coupling
        V = U - K1 @ S
        x = x1 + V @ b
        P = P1 + V @ P2 @ V.T
        P = (P + P.T)/2
        # -------------------------------------------
        self.x = x
        self.P = P
        self.x1 = x1
        self.P1 = P1
        self.b = b
        self.P2 = P2
        self.V = V
