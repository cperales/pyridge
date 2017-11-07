def accuracy(predicted_targets, real_targets):
    """
    Accuracy.

    :param predicted_targets:
    :param real_targets:
    :return:
    """
    comp = predicted_targets == real_targets
    return comp.sum() / len(comp)


def gm(something):
    pass


def ms(something):
    pass

