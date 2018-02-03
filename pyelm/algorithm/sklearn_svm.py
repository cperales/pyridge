from sklearn.svm import SVC
from pyelm.utils.preprocess import prepare_data
from pyelm.utils import accuracy
import numpy as np
from pyelm.utils import cross_validation


class SklearnSVC(SVC):
    """

    :param SVC:
    :return:
    """
    grid_param = {}

    def __call__(self, parameters=None):
        self.C = parameters['C']
        self.gamma = parameters['k']
        self.kernel = 'rbf'

    def set_cv_range(self, hyperparameters={'C': 0, 'k': 1, 'kernelFun': 'rbf'}):
        # Regularization
        self.grid_param['C'] = np.array(hyperparameters['C']) if 'C' in hyperparameters \
            else np.array([0], dtype=np.float)
        self.grid_param['k'] = np.array(hyperparameters['k']) if 'k' in hyperparameters \
            else np.array([1], dtype=np.float)

    def save_clf_param(self):
        return {'C': self.C,
                'k': self.gamma}
