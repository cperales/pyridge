import numpy as np

from pyelm.utils.target_encode import j_decode


# Metrics once classifier is fit and new data is classified


def accuracy(pred_targ, real_targ, j_encoded=True):
    """
    Percentage of predicted targets that actually
    coincide with real targets.

    :param numpy.array pred_targ: array of the targets
        according to the classifier.
    :param numpy.array real_targ: array of the real targets.
    :param bool real_targ: array of the real targets.
    :return:
    """
    if j_encoded is True:
        pred_targ = j_decode(pred_targ)
        real_targ = j_decode(real_targ)
    # comp = (pred_targ == real_targ).sum(1)
    # t = real_targ.shape[1]
    # comp[comp[:] != t] = 0
    # comp[comp[:] == t] = 1
    comp = np.array((pred_targ == real_targ), dtype=np.float)
    acc = np.mean(comp)
    return acc


def loss(predicted_targets, real_targets):
    """
    Inverse of the accuracy. It is used for cross validation.

    :param predicted_targets:
    :param real_targets:
    :return:
    """
    acc = accuracy(pred_targ=predicted_targets,
                   real_targ=real_targets)
    return 1 - acc
