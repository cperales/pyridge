import numpy as np
from sklearn.gaussian_process.kernels import RBF, DotProduct

from pyelm.generic.classifier import Classifier

kernel_fun_dict = {'rbf': RBF,
                   'linear': DotProduct}


class KernelMethod(Classifier):
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
    kernel_fun = None

    def set_range_param(self, method_conf):
        # Neuron function
        self.kernel = kernel_fun_dict[method_conf['kernelFun']]
        # Regularization
        self.grid_param['C'] = np.array(method_conf['C']) if 'C' in method_conf \
            else np.array([0], dtype=np.float)
        self.grid_param['k'] = np.array(method_conf['k']) if 'k' in method_conf \
            else np.array([1], dtype=np.float)
