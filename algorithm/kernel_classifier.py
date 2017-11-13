from .classifier import Classifier
import numpy as np


def rbf_kernel(X, Y=None, k=1.0):
    """
    Radial basis function kernel.

    :param X:
    :param Y:
    :param k:
    :return: K
    """
    Y = X if Y is None else Y
    # K = np.zeros((X.shape[0], Y.shape[1])) if len(Y.shape) > 1 else np.zeros((X.shape[0],))
    K = np.zeros((X.shape[0], Y.shape[0]))

    # # Second implementation
    # xx_h_1 = np.dot(np.matrix([np.sum(np.power(X, 2), 1)]).T, np.ones((1, Y.shape[0])))
    # xx_h_2 = np.dot(np.matrix([np.sum(np.power(Y, 2), 1)]).T, np.ones((1, X.shape[0])))
    # exponent = xx_h_1 + xx_h_2 - 2 * np.dot(X, Y.T)
    # K = np.exp(- exponent) / k

    # First implementation
    # TODO: just upper matrix
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i, j] = np.exp(-(np.linalg.norm(x - y)**2) / k)
    return K


def linear_kernel(X, Y=None):
    """

    :param X:
    :param Y:
    :return: K
    """
    Y = X if Y is None else Y
    K = np.dot(X, Y.T)
    return K


class KernelClassifier(Classifier):
    """
    Generic class for a supervised classifier with mapping function inside, besides linear.
    """
    kernel_dict = {'rbf': rbf_kernel,
                   'linear': linear_kernel}

    def __init__(self):
        pass
