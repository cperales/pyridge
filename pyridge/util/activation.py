import numpy as np
from scipy.special import expit


def sigmoid(x):
    """
    Sigmoid function. It can be replaced with scipy.special.expit.

    :param x:
    :return:
    """
    return expit(x)


def sigmoid_der(x):
    """
    Derivative of the sigmoid function.

    :param y:
    :return:
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


def leaky_relu(x, alpha=0.01):
    """
    Leaky rectified linear unit.

    :param x:
    :param float alpha: (optional) value of leak.
    :return:
    """
    return np.maximum(alpha * x, x)


def relu(x):
    """
    Rectified linear unit.

    :param x:
    :param float alpha: (optional) value of leak.
    :return:
    """
    return np.maximum(0.0, x)


def leaky_relu_der(x, alpha=0.01):
    """
    Derivative of leaky relu.

    :param x:
    :param float alpha: (optional) value of leak.
    :return:
    """
    y = np.ones_like(x)
    y[x > 0] = alpha
    return y


def tanh(x):
    """
    Hyperbolic tangent

    :param x:
    :return:
    """
    return np.tanh(x)


def arctan(x):
    """
    Tan^-1
    :param x:
    :return:
    """
    return np.arctan(x)


def tanh_der(x):
    """
    Derivative of the hyperbolic tangent function.

    :param x:
    :return:
    """
    return 1.0 - np.power(tanh(x), 2)


def linear(x):
    """
    Linear function.

    :param x:
    :return:
    """
    return x


def linear_der(x):
    """
    Derivate of the linear function.

    :param x:
    :return:
    """
    return 1.0


def soft_plus(x):
    """
    Soft plus function.

    :param x:
    :return:
    """
    return np.log(1.0 + np.exp(x))


def soft_plus_der(x):
    """
    Soft plus function.

    :param x:
    :return:
    """
    return np.power(1.0 + np.exp(-x), -1)


def selu(x, lambda_=1.0507, alpha=1.67326):
    """
    Scaled exponential linear unit.

    :param x:
    :param float lambda_:
    :param float alpha:
    :return:
    """
    a = x
    a[x < 0.0] = alpha * (np.exp(a[x < 0.0]) - 1.0)
    return lambda_ * a


def sinc(x):
    """
    Sinc function.

    :param x:
    :return:
    """
    return np.sinc(x)


def gaussian(x):
    """

    :param x:
    :return:
    """
    return np.exp(- np.power(x, 2))


# Activation dict for Neural Network with their derivatives
nn_activation_dict = {
    'sigmoid': {'activation': sigmoid,
                'derivative': sigmoid_der},
    'sin': {'activation': np.sin,
            'derivative': np.cos},
    # 'leaky_relu': {'activation': leaky_relu,  # It works bad
    #                'derivative': leaky_relu_der},
    'tanh': {'activation': tanh,
             'derivative': tanh_der},
    'linear': {'activation': linear,
               'derivative': linear_der},
    'soft_plus': {'activation': soft_plus,
                  'derivative': soft_plus_der}
}

# Activation dict for ELM
activation_dict = {
    'sin': np.sin,
    'cos': np.cos,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'hard': lambda x: np.array(x > 0.0, dtype=float),
    'linear': linear,
    'tanh': tanh,
    'sigmoid': sigmoid,
    'sinc': sinc,
    'gaussian': gaussian,
    'selu': selu,
    'arctan': arctan,
    'soft_plus': soft_plus,
}


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
