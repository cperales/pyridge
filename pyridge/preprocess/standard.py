from pyridge.generic.scaler import Scaler
import numpy as np


class StandardScaler(Scaler):
    """
    Scaler for data, similar to StandardScaler from
    sklearn but avoiding shape restrictions.
    """
    def __init__(self):
        self.mean_: np.float
        self.std_: np.float

    def get_params(self):
        return {'mean_': self.mean_, 'std_': self.std_}

    def fit(self, values):
        self.mean_ = np.mean(values, axis=0)
        self.std_ = np.std(values, axis=0)

    def transform(self, values):
        return (values - self.mean_) / self.std_

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * self.std_ + self.mean_