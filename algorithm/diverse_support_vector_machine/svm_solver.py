import numpy as np
import pickle
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


COLORS = ['red', 'blue']
solvers.options['show_progress'] = False


def read_data(f):
    """
    Function that read a pickle.

    :param f: filename of the data.
    :return: x, y
    """
    with open(f, 'rb') as f:
        data = pickle.load(f)
    x, y = data[0], data[1]
    return x, y


def fit(x, y):
    """
    Fit alphas for dual problem hard margin SVM.

    :param x: instances.
    :param y: labels.
    :return: alphas
    """
    num = x.shape[0]
    dim = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    # Same that np.dot(np.dot(np.eye(num), K), K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    G = matrix(-np.eye(num))
    h = matrix(np.zeros(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def fit_soft(x, y, c=1.0):
    """
    Fit alphas for dual problem soft margin SVM.

    :param x: instances.
    :param y: labels.
    :return: alphas
    """
    num = x.shape[0]
    dim = x.shape[1]
    C = c / dim  # Penalty
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    g = np.concatenate((-np.eye(num), np.eye(num)))
    G = matrix(g)
    h_array = np.concatenate((np.zeros(num), C * np.ones(num)))
    h = matrix(h_array)
    # G = matrix(np.eye(num))
    # h = matrix(np.ones(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def fit_diverse(x, y, w, c=1.0, d=1.0):
    """
    Fit alphas for dual problem soft margin diverse SVM.

    :param x: instances.
    :param y: labels.
    :param w: vector of previous SVM.
    :param c:
    :param d:
    :return: alphas
    """
    # Normalize w
    u = w / np.sqrt(np.dot(w, w))

    num = x.shape[0]
    dim = x.shape[1]
    C = c / dim  # Penalty
    D = d  # Difference with previous SVM
    # we'll solve the dual
    # obtain the kernel
    new_x = np.array([np.array([elem[0] * u[0] * D, elem[1] * u[1] * D]) for elem in x])
    K = y[:, None] * new_x
    I = np.eye(num)
    K = np.dot(np.dot(I, K), K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    # G = matrix(-np.eye(num))
    # h = matrix(np.zeros(num))
    g = np.concatenate((-I, I))
    G = matrix(g)
    h_array = np.concatenate((np.zeros(num), C * np.ones(num)))
    h = matrix(h_array)
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def plot_data_with_labels(x, y, ax):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])


def plot_separator(ax, w, b, color='k'):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(0, 6)
    ax.plot(x, x * slope + intercept, color + '-')


if __name__ == '__main__':
    # # # DIVERSE SVM
    # x, y = read_data('gaussiandata.pickle')
    x, y = read_data('gaussiandata_soft.pickle')
    # x, y = read_data(sys.argv[1])

    color = 'k'  # Black
    # fit svm classifier
    # alphas = fit(x, y)
    alphas = fit_soft(x, y)

    # get weights
    w = np.sum(alphas * y[:, None] * x, axis = 0)
    # get bias
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]

    # normalize
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm

    # show data and w
    fig, ax = plt.subplots()
    plot_separator(ax, w, bias, color=color)
    plot_data_with_labels(x, y, ax)
    # plt.show()

    # # DIVERSE
    # x, y = read_data(sys.argv[1])
    # fit svm classifier
    alphas = fit_diverse(x, y, w)

    # get weights
    w = np.sum(alphas * y[:, None] * x, axis = 0)
    # get bias
    cond = (alphas > 1e-4).reshape(-1)
    b = y[cond] - np.dot(x[cond], w)
    bias = b[0]

    # normalize
    norm = np.linalg.norm(w)
    w, bias = w / norm, bias / norm

    # show data and w
    # fig, ax = plt.subplots()
    plot_separator(ax, w, bias, color='b')
    plot_data_with_labels(x, y, ax)
    plt.show()

    # # # SOFT MARGIN
    # x, y = read_data('gaussiandata_soft.pickle')
    # # x, y = read_data(sys.argv[1])
    #
    # color = 'k'  # Black
    # # fit svm classifier
    # alphas = fit_soft(x, y)
    #
    # # get weights
    # w = np.sum(alphas * y[:, None] * x, axis=0)
    # # get bias
    # cond = (alphas > 1e-4).reshape(-1)
    # b = y[cond] - np.dot(x[cond], w)
    # bias = b[0]
    #
    # # normalize
    # norm = np.linalg.norm(w)
    # w, bias = w / norm, bias / norm
    #
    # # show data and w
    # fig, ax = plt.subplots()
    # plot_separator(ax, w, bias, color=color)
    # plot_data_with_labels(x, y, ax)
    # plt.show()