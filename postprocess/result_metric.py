import numpy as np
# Metrics once classifier is fit and new data is classified


def accuracy(predicted_targets, real_targets):
    """
    Accuracy.

    :param predicted_targets:
    :param real_targets:
    :return:
    """
    comp = np.array([e[0] == e[1] for e in zip(predicted_targets, real_targets)], dtype=bool)
    # comp = predicted_targets[:, None] == real_targets[:, None]
    acc = np.mean(comp)
    return acc
