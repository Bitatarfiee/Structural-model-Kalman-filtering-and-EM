# Linear Gaussian State Space model

import numpy as np
from scipy.stats import norm


class LGSS:
    """Linear Gaussian State Space model. The observation is assumed to be one-dimensional."""

    def __init__(self, T, R, Q, Z, H, a1, P1):
        self.d = T.shape[0]  # State dimension
        self.deta = R.shape[1]  # Second dimension is process noise dim
        self.T = T  # Process model
        self.R = R  # Process noise prefactor
        self.Q = Q  # Process noise covariance
        self.Z = Z  # Measurement model
        self.H = H  # Measurement noise variance
        self.a1 = a1  # Initial state mean
        self.P1 = P1  # Initial state covariance

    def get_params(self):
        """Return all model parameters.

        T, R, Q, Z, H, a1, P1 = model.get_params()
        """
        return self.T, self.R, self.Q, self.Z, self.H, self.a1, self.P1