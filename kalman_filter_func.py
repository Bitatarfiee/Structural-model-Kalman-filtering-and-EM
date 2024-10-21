# kalman filter code

import numpy as np
from scipy.stats import norm


class kfs_res:
    """Container class to store result of Kalman filter and smoother."""

    def __init__(self, alpha_pred, P_pred, alpha_filt, P_filt, y_pred, F_pred):
        """Initialize with KF results"""
        self.alpha_pred = alpha_pred
        self.P_pred = P_pred
        self.alpha_filt = alpha_filt
        self.P_filt = P_filt
        self.y_pred = y_pred
        self.F_pred = F_pred

    def set_ks_res(self, alpha_sm, V, eps_hat, eps_var, eta_hat, eta_cov):
        """Update to contain also KS results"""
        self.alpha_sm = alpha_sm
        self.V = V
        self.eps_hat = eps_hat
        self.eps_var = eps_var
        self.eta_hat = eta_hat
        self.eta_cov = eta_cov

def kalman_filter(y, model: LGSS):
    """Kalman filter for LGSS model with one-dimensional observation.

    :param y: (n,) array of observations. May contain nan, which encodes missing observations.
    :param model: LGSS object with the model specification.
    
    :return kfs_res: Container class with member variables,
        alpha_pred: (d,1,n) array of predicted state means.
        P_pred: (d,d,n) array of predicted state covariances.
        alpha_filt: (d,1,n) array of filtered state means.
        P_filt: (d,d,n) array of filtered state covariances.
        y_pred: (n,) array of means of p(y_t | y_{1:t-1})
        F_pred: (n,) array of variances of p(y_t | y_{1:t-1})
    """

    n = len(y)
    d = model.d  # State dimension
    alpha_pred = np.zeros((d, 1, n))
    P_pred = np.zeros((d, d, n))
    alpha_filt = np.zeros((d, 1, n))
    P_filt = np.zeros((d, d, n))
    y_pred = np.zeros(n)
    F_pred = np.zeros(n)

    T, R, Q, Z, H, a1, P1 = model.get_params()  # Get all model parameters (for brevity)

    for t in range(n):
        # Time update (predict)
        if t == 0:
            alpha_pred[:, :, t] = a1
            P_pred[:, :, t] = P1
        else:
            alpha_pred[:, :, t] = np.dot(T, alpha_filt[:, :, t - 1])
            P_pred[:, :, t] = np.dot(np.dot(T, P_filt[:, :, t - 1]), T.T) + np.dot(np.dot(R, Q), R.T)
        
        # Compute prediction of current output
        y_pred[t] = np.dot(Z, alpha_pred[:, :, t])
        F_pred[t] = np.dot(np.dot(Z, P_pred[:, :, t]), Z.T) + H
        
        # Measurement update (correct)
        if np.isnan(y[t]):
            alpha_filt[:, :, t] = alpha_pred[:, :, t]
            P_filt[:, :, t] = P_pred[:, :, t]
            
        else:
            K_t = np.dot(P_pred[:, :, t], Z.T)/F_pred[t]
            alpha_filt[:, :, t] = alpha_pred[:, :, t] + K_t*(y[t] - y_pred[t])
            P_filt[:, :, t] = P_pred[:, :, t] - np.dot(np.dot(K_t, Z), P_pred[:, :, t])
        
        
        
    kf = kfs_res(alpha_pred, P_pred, alpha_filt, P_filt, y_pred, F_pred)
    return kf