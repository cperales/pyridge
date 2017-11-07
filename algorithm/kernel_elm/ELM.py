from ..kernel_classifier import KernelClassifier
from scipy.sparse import identity
import numpy as np


class KernelELM(KernelClassifier):
    """
    Kernel version of Extreme Learning Machine classifier.
    """
    def __init__(self, **kwargs):
        self.c = kwargs['c'] if 'c' in kwargs else 1.0
        self.d = kwargs['k'] if 'k' in kwargs else 1.0
        kernel_type = kwargs['kernel_type'] if 'kernel_type' in kwargs else 'linear'
        self.kernel_function = self.kernel[kernel_type]
        # Data
        self.train_data = None
        self.train_target = None
        # Parameters
        self.beta = None
        self.k = None

    def fit(self, train_data, train_target, k=1.0):
        """

        :param train_data:
        :param train_target:
        :return:
        """
        self.k = 1.0
        self.train_data = train_data
        self.train_target = train_target
        n = self.train_data.shape[0]
        omega_train = self.kernel_function(X=self.train_data, Y=self.train_data, k=self.k)
        beta = np.linalg.solve((omega_train + identity(n) / self.c), self.train_target)
        self.beta = beta

    def classify(self, data):
        """

        :param data:
        :return:
        """
        omega_clf = self.kernel_function(X=self.train_data, Y=data, k=self.C)
        indicator = np.dot(omega_clf.T, self.beta)
