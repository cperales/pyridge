# from sklearn.gaussian_process.kernels import RBF, DotProduct
from ..util.activation import kernel_dict
from .predictor import Predictor


class KernelMethod(Predictor):
    __name__ = 'Kernel classifier'
    gamma: float
    kernel: str
    kernel_fun = None
