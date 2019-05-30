import numpy as np
from math import sqrt

class GaussianMeanShiftRegistration:
    def __init__(self, x, P, F, Q, H, hx, R):
        self.x = np.array(x)
        self.P = P
        self.b = np.array([0., 0.])
        self.F = F
        self.Q = Q
        self.H = H
        self.hx = hx
        self.R = R

        self.residual_set = []
        self.b_conv = np.array([0., 0.])

    def predict_and_update(self, dt, z):
        x = self.x
        P = self.P
        F = self.F
        H = self.H
        hx = self.hx
        b = self.b
        # use EKF to estimate state
        x_prior = F(x, dt) @ x
        P_prior = F(x, dt) @ P @ F(x, dt).T + self.Q
        S = H(x, dt) @ P_prior @ H(x, dt).T + self.R
        K = P_prior @ H(x, dt).T @ np.linalg.inv(S)
        x = x_prior + K @ (z - hx(x_prior, dt) - b)
        # use estimated state to predict observation
        z_predict = hx(x, dt) + b
        # generate observation set
        residual = z - z_predict
        if len(self.residual_set) < 3:
            self.residual_set.append(residual)
        else:
            self.residual_set[0] = self.residual_set[1]
            self.residual_set[1] = self.residual_set[2]
            self.residual_set[2] = residual
        # points shift
        thre = np.array([1.,1.])
        while sqrt(thre[0]**2+thre[1]**2)>0.001:
            Gs = [(self.b_conv - residual) @ np.linalg.inv(self.R) @ (self.b_conv - residual) for residual in self.residual_set]
            norms = [sqrt(residual[0]**2 + residual[1]**2) for residual in self.residual_set]
            M = sum(norms)
            M =1
            num = [np.exp(-0.5 * Gs[j]/M) * self.residual_set[j] for j in range(len(self.residual_set))]
            num = sum(num)
            den = [np.exp(-0.5 * Gs[j]/M) for j in range(len(self.residual_set))]
            den = sum(den)
            b_conv_new = num/den
            thre = b_conv_new - self.b_conv
            self.b_conv = b_conv_new
        
        
        self.x = x
        self.P = P
        self.b = b
