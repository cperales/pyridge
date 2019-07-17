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
