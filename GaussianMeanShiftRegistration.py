import numpy as np
from math import sqrt
from KalmanFilter import KalmanFilter

class GaussianMeanShiftRegistration:
    def __init__(self, x, P, Q, R, F, H, hx):
        """
        :param x: list
        :param P: list, sqrt of diagnose of P, std of each variable
        :param Q: 2d ndarray
        :param R: list, sqrt of diagnose of R, std of each measurement
        :param F: (x, dt) => 2d ndarray
        :param H: (x) => 2d ndarray
        :param hx: (x) => 1d ndarray
        """
        self.R = np.diag(R)**2
        self.hx = hx
        self.b = np.zeros((len(R),))

        self.ekf = KalmanFilter(x, P, Q, R, F=F, H=H, 
            hx=lambda x,b: hx(x) + b)

        self.residual_set = []
        self.b_conv = np.array([0., 0.])

    def predict_and_update(self, dt, z, **args):
        b = self.b
        # use EKF to estimate state
        self.ekf.predict_update(dt, z, b=self.b)
        # use estimated state to predict observation
        z_predict = self.hx(self.ekf.x) 
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
        while sqrt(thre[0]**2+thre[1]**2)<0.00001:
            Gs = [(self.b_conv - residual) @ np.linalg.inv(self.R) @ (self.b_conv - residual) for residual in self.residual_set]
            norms = [sqrt(residual[0]**2 + residual[1]**2) for residual in self.residual_set]
            M = sum(norms)
            num = [np.exp(-0.5 * Gs[j]/M) * self.residual_set[j] for j in range(len(self.residual_set))]
            num = sum(num)
            den = [np.exp(-0.5 * Gs[j]/M) for j in range(len(self.residual_set))]
            den = sum(den)
            b_conv_new = num/den
            thre = b_conv_new - self.b_conv
            self.b_conv = b_conv_new
        
        
        # self.b = self.b_conv.copy()
