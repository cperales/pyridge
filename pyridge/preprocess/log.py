import numpy as np
from pyelm.generic.scaler import Scaler


class LogScaler(Scaler):
    """
    Scaler for that transform the values in a logaritmic
    scaler.
    """
    def __init__(self):
        self.min_: np.float

    def get_params(self):
        return {'min_': self.min_, 'max_': self.max_}

    def fit(self, values):
        self.min_ = np.min(values, axis=0)

    def transform(self, values):
        return np.log(values + (self.min_ + 1.0))

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return np.exp(values) - (self.min_ + 1.0)
