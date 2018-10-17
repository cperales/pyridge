import numpy as np

def linear_kernel(gamma: float = 1.0, X=None, Y=None):
    """
    Linear kernel (dot product).

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
    Radial Basis Function kernel.

    :param float gamma:
    :param np.matrix X:
    :param Y:
    :return:
    """
    omega_linear = linear_kernel(X=X, Y=Y)
    omega = np.exp(- omega_linear / gamma, dtype=np.float64)
    return omega
