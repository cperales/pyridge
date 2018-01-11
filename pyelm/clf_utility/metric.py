import numpy as np

from pyelm.clf_utility.target_encode import j_decode


# Metrics once classifier is fit and new data is classified


def accuracy(predicted_targets, real_targets):
    """
    Simple accuracy, number of predicted targets that actually
    coincide with real targets

    :param predicted_targets: array of the targets according to the classifier.
    :param real_targets: array of the real targets.
    :return:
    """
    pred_targ = j_decode(predicted_targets)
    real_targ = j_decode(real_targets)
    # comp = (predicted_targets == real_targets).sum(1)
    # t = real_targets.shape[1]
    # comp[comp[:] != t] = 0
    # comp[comp[:] == t] = 1
    comp = np.array((pred_targ == real_targ), dtype=np.float)
    acc = np.mean(comp)
    return acc


def loss(predicted_targets, real_targets):
    """
    Invers of the accuracy. It is used for cross validation.

    :param predicted_targets:
    :param real_targets:
    :return:
    """
    acc = accuracy(predicted_targets=predicted_targets,
                   real_targets=real_targets)
    return 1 - acc
