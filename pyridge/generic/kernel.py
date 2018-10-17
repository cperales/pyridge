# from sklearn.gaussian_process.kernels import RBF, DotProduct
from pyridge.util.kernel import *
from .classifier import Classifier

kernel_dict = {'rbf': rbf_kernel,
               'linear': linear_kernel}


class KernelMethod(Classifier):
    __name__ = 'Kernel classifier'
    gamma: float = 1.0
    kernel: str = 'rbf'
    kernel_fun = None
