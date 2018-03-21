import numpy as np
from sklearn.gaussian_process.kernels import RBF, DotProduct

from pyelm.generic.classifier import Classifier

kernel_fun_dict = {'rbf': RBF,
                   'linear': DotProduct}


class KernelMethod(Classifier):
    __name__ = 'Kernel classifier'
    # Cross validated parameters
    C = 0
    grid_param = {'C': C}
    k = 1

    # Neural network features
    output_weight = 0
    t = 2  # At least, 2 labels are classified

    # Kernel
    train_data = None
    kernel = None
    kernel_fun = RBF(length_scale=k)  # By default

    def set_cv_range(self,
                     hyperparameters={'C': 0, 'k': 1, 'kernelFun': 'rbf'}):
        # Neuron function
        self.kernel = kernel_fun_dict[hyperparameters['kernelFun']]
        # Regularization
        self.grid_param['C'] = np.array(hyperparameters['C']) if \
            'C' in hyperparameters \
            else np.array([0], dtype=np.float)
        self.grid_param['k'] = np.array(hyperparameters['k']) if \
            'k' in hyperparameters \
            else np.array([1], dtype=np.float)
