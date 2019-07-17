import numpy as np
from scipy.special import expit


def sigmoid(x):
    """
    Sigmoid function. It can be replaced with scipy.special.expit.

    :param x:
    :return:
    """
    return expit(x)


def sigmoid_derivative(y):
    """
    Derivate of the sigmoid function.
    We assume y is already sigmoided.

    :param y:
    :return:
    """
    return y * (1.0 - y)


activation_dict = {'sin': np.sin,
                   'relu': lambda x: np.maximum(x, 0.0),
                   'hard': lambda x: np.array(x > 0.0, dtype=float),
                   # 'sigmoid': lambda x: 1.0/(1.0 + np.exp(-x))
                   'sigmoid': sigmoid}  # Faster for matrices


def linear_kernel(gamma: float = 1.0, X=None, Y=None):
    """

    :param gamma:
    :param X:
    :param Y:
    :return:
    """
    n = X.shape[0]
    X = gamma * X
    if Y is None:
        XXh = np.dot(np.sum(X**2, 1, dtype=np.float64).reshape((n, 1)), np.ones((1, n), dtype=np.float64))
        omega = XXh + XXh.transpose() - 2.0 * np.dot(X, X.transpose())
    else:
        m = Y.shape[0]
        XXh = np.dot(np.sum(X**2, 1, dtype=np.float64).reshape((n, 1)), np.ones((1, m), dtype=np.float64))
        YYh = np.dot(np.sum(Y**2, 1, dtype=np.float64).reshape((m, 1)), np.ones((1, n), dtype=np.float64))
        omega = XXh + YYh.transpose() - 2.0 * np.dot(X, Y.transpose())
    return np.array(omega, dtype=np.float64)


def rbf_kernel(gamma: float = 1.0, X=None, Y=None):
    """
    Function to obtain omega matrix.

    :param float gamma:
    :param np.matrix X:
    :param Y:
    :return:
    """
    omega_linear = linear_kernel(X=X, Y=Y)
    omega = np.exp(- omega_linear / gamma, dtype=np.float64)
    return omega


def u_dot_norm(u):
    """
    Return u vector with norm = 1.

    :param u:
    :return:
    """
    # return u / np.sqrt(np.dot(u.T, u))
    return u / np.dot(u.T, u)


kernel_dict = {'rbf': rbf_kernel,
               'linear': linear_kernel}
