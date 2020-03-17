import numpy as np
import scipy.linalg as la
from cvxopt import lapack, matrix


def np_solve(a, b):
    """
    Inverse of a generic matrix.

    :param a:
    :param b:
    :return:
    """
    x_solve = np.linalg.solve(a=a, b=b)
    return x_solve


def sp_solve(a, b):
    """
    Inverse of a symmetric matrix.

    :param a:
    :param b:
    :return:
    """
    x_sp_solve = la.solve(a=a,
                          b=b,
                          lower=False,
                          overwrite_a=True,
                          overwrite_b=True,
                          debug=None,
                          check_finite=False,
                          transposed=False,
                          assume_a='sym')
    return x_sp_solve


def lapack_zhesv(a, b):
    """
    Inverse matrix usando LAPACK zhesv.

    :param a:
    :param b:
    :return:
    """
    x_zhesv = np.real(la.lapack.zhesv(a=a, b=b)[2])
    return x_zhesv


def lapack_posv(a, b):
    """
    Inverse using LaPaCK posv (definite positive) algorithm.

    :param a:
    :param b:
    :return:
    """
    b_posv = matrix(b)
    lapack.posv(matrix(a), b_posv)
    return b_posv


def lapack_hesv(a, b):
    """
    Inverse using LaPaCK hesv (hermitian) algorithm.

    :param a:
    :param b:
    :return:
    """
    b_her = matrix(b)
    lapack.hesv(matrix(a), b_her)
    return b_her


def lapack_potrs(a, b):
    """
    Inverse using LaPaCK potrs (hermitian) algorithm.

    :param a:
    :param b:
    :return:
    """
    b_her = matrix(b)
    lapack.potrs(matrix(a), b_her)
    return b_her


def lapack_sysv(a, b):
    """
    Inverse using LaPaCK sysv algorithm.

    :param a:
    :param b:
    :return:
    """
    b_sysv = matrix(b)
    lapack.sysv(matrix(a), b_sysv)
    return b_sysv


def np_solve_inv(a, b):
    """
    SLOW. Pure inverse.

    :param a:
    :param b:
    :return:
    """
    x_solve_inv = np.dot(np.linalg.inv(a), b)
    return x_solve_inv


def np_solve_pinv(a, b):
    """
    REALLY SLOW. Pinv inverse.

    :param a:
    :param b:
    :return:
    """
    x_solve_pinv = np.dot(np.linalg.pinv(a), b)
    return x_solve_pinv


def cov_n(h):
    """
    Personal implementation of covariance matrix,
    which result is np.cov(h) * ( N_j - 1 )

    :param matrix:
    :return:
    """
    final_dim = h.shape[1]
    cov_matrix = np.empty((final_dim, final_dim))
    for row in range(final_dim):
        for column in range(final_dim):
            if row == column:  # Diag
                h_d = h[:, row]
                value = np.sum(np.power(h_d - np.mean(h_d), 2))
            else:
                h_d = h[:, row]
                h_i = h[:, column]
                value = np.sum((h_d - np.mean(h_d)) * (h_i - np.mean(h_i)))
            cov_matrix[row, column] = value
    return cov_matrix


def cov_pen(h_j, h_no_j):
    """
    Personal implementation of covariance matrix with penalization.

    :param h_j:
    :param h_no_j:
    :return:
    """
    final_dim = h_j.shape[1]
    cov_matrix = np.empty((final_dim, final_dim))
    for row in range(final_dim):
        for column in range(final_dim):
            h_d = h_j[:, row]
            h_d_no_j = h_no_j[:, row]
            a = h_d - np.mean(h_d)
            if row == column:  # Diag
                value = np.dot(a.T, a) + np.dot(h_d_no_j.T, h_d_no_j)
            else:
                h_i = h_j[:, column]
                h_i_no_j = h_no_j[:, column]
                b = h_i - np.mean(h_i)
                value = np.dot(a.T, b) + np.dot(h_d_no_j.T, h_i_no_j)
            cov_matrix[row, column] = value
    return cov_matrix


def check_symmetric(matrix):
    """

    :param matrix:
    :return:
    """
    return np.isclose(matrix, matrix.T).all()


def get_matrix_j(h_j):
    """

    :param h_j:
    :param h_no_j:
    :return:
    """
    final_dim = h_j.shape[1]
    cov_matrix = np.empty((final_dim, final_dim))
    for row in range(final_dim):
        for column in range(final_dim):
            if row == column:  # Diag
                h_d = h_j[:, row]
                value = np.sum(np.power(h_d - np.mean(h_d), 2))
            else:
                h_d = h_j[:, row]
                h_i = h_j[:, column]
                value = np.sum((h_d - np.mean(h_d)) * (h_i - np.mean(h_i)))
            cov_matrix[row, column] = value
    return cov_matrix


def get_matrix_no_j(h_no_j):
    """

    :param h_no_j:
    :return:
    """
    final_dim = h_no_j.shape[1]
    matrix = np.empty((final_dim, final_dim))
    for row in range(final_dim):
        if row == column:  # Diag
            h_d = h_j[:, row]
            value = np.sum(h_d * h_d)
        else:
            h_d = h_j[:, row]
            h_i = h_j[:, column]
            value = np.sum(h_d * h_i)
        matrix[row, column] = value
    return matrix
