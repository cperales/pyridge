from pyridge.generic.scaler import Scaler
import numpy as np


class MinMaxScaler(Scaler):
    """
    Scaler for target, similar to MinMaxScaler from
    sklearn but avoiding shape restrictions.
    """
    def __init__(self):
        self.min_: np.float
        self.max_: np.float

    def get_params(self):
        return {'min_': self.min_, 'max_': self.max_}

    def fit(self, values):
        self.min_ = values.min()
        self.max_ = values.max()

    def transform(self, values):
        return (values - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, values):
        return values * (self.max_ - self.min_) + self.min_