import numpy as np


def j_renorm(T):
    idx = T.argmax(axis=1)
    out = np.zeros_like(T, dtype=float)
    out[np.arange(T.shape[0]), idx] = 1
    return out


def j_encode(targ, n_targ=None):
    n = len(targ)
    if n_targ is None:  # No information provided
        ava_targ = np.unique(targ)
        n_targ = len(ava_targ)
    else:
        ava_targ = np.arange(1, n_targ)

    T = np.zeros((n, n_targ))
    for i in range(n):
        row = [targ[i] == ava_targ]
        T[i] = np.array(row, dtype=np.float)

    return T


def j_decode(T):
    pass
