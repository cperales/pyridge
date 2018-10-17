import numpy as np


def accuracy(clf, pred_data, real_targ):
    """
    Percentage of predicted targets that actually
    coincide with real targets.

    :param clf: classifier with predict method.
    :param pred_data: array of the targets
        according to the classifier.
    :param numpy.array real_targ: array of the real targets.
    :param bool real_targ: array of the real targets.
    :return:
    """
    pred_targ = clf.predict(pred_data)
    comp = np.array((pred_targ == real_targ), dtype=np.float64)
    acc = np.mean(comp)
    return acc


def loss(predicted_targets, real_targets):
    """
    Inverse of the accuracy. It is used for cross validation.

    :param predicted_targets:
    :param real_targets:
    :return:
    """
    comp = np.array((predicted_targets == real_targets),
                    dtype=np.float64)
    acc = np.mean(comp)
    return 1 - acc


def rmse(clf, pred_data, real_targ):
    """

    :param clf:
    :param pred_data:
    :param real_targ:
    :return:
    """
    real_j_targ = clf.Y
    ind_matrix = clf.get_indicator(pred_data)

    rmse_vec = [np.linalg.norm(real_row - ind_row)
                for ind_row, real_row in zip(ind_matrix,
                                             real_j_targ)]
    return np.mean(rmse_vec)


def diversity(clf, pred_data, real_targ):
    """
    Implemented directly from MATLAB, not pythonic.
    TODO: rewrite.

    :param clf: Classifier.
    :param pred_data: Not used.
    :param real_targ: Not used.
    :return:
    """
    count = 1
    div = 0.0
    for s in range(clf.size):
        beta_s = clf.output_weight[s]
        for t in range(s + 1, clf.size):
            beta_t = clf.output_weight[t]
            for j in range(beta_t.shape[1]):
                beta_s_j = beta_s[:, j]
                beta_t_j = beta_t[:, j]
                num = np.dot(np.dot(beta_s_j.T, beta_t_j),
                                      np.dot(beta_t_j.T, beta_s_j))
                dem = np.dot(np.dot(beta_t_j.T, beta_t_j),
                             np.dot(beta_s_j.T, beta_s_j))
                div_step = 1 - num / dem
                div += div_step
                count += 1
    return div / float(count)


def kernel_diversity(clf, pred_data, real_targ):
    """
    TODO: Implement.

    :param clf: Classifier
    :param pred_data: Not used.
    :param real_targ: Not used.
    :return:
    """
    pass
