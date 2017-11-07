from .classifier import Classifier
import numpy as np


def rbf_kernel(X, Y, C):
    """
    Radial basis function kernel.

    :param X:
    :param Y:
    :param C:
    :return: K
    """
    K = np.zeros((X.shape[0],Y.shape[0]))
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i, j] = np.exp(-(np.linalg.norm(x - y)**2) / C)
    return K


def linear_kernel(X, Y):
    """

    :param X:
    :param Y:
    :return: K
    """
    K = np.dot(X, Y.T)
    return K


class KernelClassifier(Classifier):
    """
    Generic class for a supervised classifier with mapping function inside, besides linear.
    """
    def __init__(self):
        pass
