import numpy as np


def rbf_kernel(gamma: float, X, Y=None):
    """
    Function to obtain omega matrix.

    :param float gamma:
    :param np.matrix X:
    :param Y:
    :return:
    """
    n = X.shape[0]
    if Y is None:
        XXh = np.dot(np.sum(X**2, 1, dtype=np.float64).reshape((n, 1)), np.ones((1, n), dtype=np.float64))
        omega = XXh + XXh.transpose() - 2.0 * np.dot(X, X.transpose())
    else:
        m = Y.shape[0]
        XXh = np.dot(np.sum(X**2, 1, dtype=np.float64).reshape((n, 1)), np.ones((1, m), dtype=np.float64))
        YYh = np.dot(np.sum(Y**2, 1, dtype=np.float64).reshape((m, 1)), np.ones((1, n), dtype=np.float64))
        omega = XXh + YYh.transpose() - 2.0 * np.dot(X, Y.transpose())
    omega = np.exp(- omega / gamma, dtype=np.float64)
    return omega


def linear_kernel(gamma, X, Y=None):
    """

    :param gamma:
    :param X:
    :param Y:
    :return:
    """
    n = X.shape[0]
    if Y is None:
        XXh = np.dot(np.sum(X**2, 1), np.ones(1, n))
        omega = XXh + XXh.transpose() - 2.0 * np.dot(X, X.transpose())
    else:
        m = Y.shape[0]
        XXh = np.dot(np.sum(X**2, 1), np.ones(1, m))
        YYh = np.dot(np.sum(Y**2, 1), np.ones(1, n))
        omega = XXh + YYh.transpose() - 2.0 * gamma * np.dot(X, Y)
    return omega
