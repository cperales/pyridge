import numpy as np


def j_renorm(T):
    idx = T.argmax(axis=1)
    out = np.zeros_like(T, dtype=float)
    out[np.arange(T.shape[0]), idx] = 1
    return out
