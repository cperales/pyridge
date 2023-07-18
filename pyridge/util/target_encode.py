import numpy as np


def id_encoder(T):
    """
    Function for linear, where there is no encoding, and
    vectors are reshaped.

    :param T: array of the targets.
    """
    if len(T.shape) == 1:
        return np.array(T, dtype=float).reshape(T.shape[0], 1)
    else:
        return T


def id_decoder(T):
    """
    Function for linear, where there is no encoding, and
    vectors are raveled

    :param T:
    :return:
    """
    return np.array(T, dtype=float).ravel()


"""
From here until the end it's DEPRECATED. Use sklearn instead.
"""


def j_renorm(T):
    """
    The maximum component of each row is filled with 1
    and the rest with 0.

    :param T: matrix of the targets
    :return:
    """
    idx = T.argmax(axis=1)
    out = np.zeros_like(T, dtype=np.float64)
    out[np.arange(T.shape[0]), idx] = 1
    return out


def j_encode(targets, n_targ=None):
    """
    Ridge classifier needs the targets are vectors: as many components
    as possible targets. The position of the vector which coincide with
    the number of the target is filled with 1, and the rest of the components
    is 0.

    For example, if the target is binary, 1 and 2, the target for an instance
    is a vector [1, 0] if the target is 1, and [0, 1] is the target is 2.

    :param targets: the array of targets.
    :param n_targ: number of different targets.
        It reduce the number of operations.
    :return T: matrix of targets.
    """
    n = len(targets)
    if n_targ is None:  # No information provided
        ava_targ = np.unique(targets)
        n_targ = len(ava_targ)
    else:
        ava_targ = np.arange(1, n_targ + 1)

    T = np.zeros((n, n_targ), dtype=np.float64)
    for i in range(n):
        row = [targets[i] == ava_targ]
        T[i] = np.array(row, dtype=np.float64)

    return T


def j_decode(T):
    """
    Transform from a matrix to an array of targets
    with normal encoding.

    :param T: matrix of targets.
    :return targets: array of targets.
    """
    if len(T.shape) < 2:  # Already decoded
        targets = T
    else:
        targets = np.argmax(T, axis=1) + 1
    return targets
