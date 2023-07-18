from pyridge.generic.scaler import Scaler
import numpy as np


class LogScaler(Scaler):
    """
    Scaler for that transform the values in a logaritmic
    scaler.
    """
    def __init__(self):
        self.min_: float

    def get_params(self):
        return {'min_': self.min_}

    def fit(self, values):
        self.min_ = np.min(values, axis=0)

    def transform(self, values):
        return np.log(values + (1.0 - self.min_))

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return np.exp(values) - (1.0 - self.min_)
