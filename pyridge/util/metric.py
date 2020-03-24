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
    comp = np.array((pred_targ == real_targ), dtype=np.float)
    acc = np.mean(comp)
    return acc


def rmse(clf, pred_data, real_targ):
    """

    :param clf:
    :param pred_data:
    :param real_targ:
    :return:
    """
    real_j_targ = clf.label_encoder(real_targ)
    ind_matrix = clf.get_indicator(pred_data)

    rmse_vec = [np.linalg.norm(real_row - ind_row)
                for ind_row, real_row in
                zip(ind_matrix, real_j_targ)]
    return np.mean(rmse_vec)


def disagreement(clf, pred_data, real_targ, S):
    """
    For Bagging Stepwise ELM Ensemble.

    :param clf: classifier
    :param pred_data: data to predict.
    :param real_targ:
    :param S:
    :return:
    """
    acc_matrix = np.empty((pred_data.shape[0], S))
    for s in range(S):
        prediction_s = clf.predict(pred_data, s).ravel()
        acc_matrix[:, s] = np.array((prediction_s == real_targ), dtype=np.float)
    Q = np.empty((S, S))
    for i in range(S - 1):
        for j in range(i + 1, S):
            c_1 = acc_matrix[:, i]
            c_2 = acc_matrix[:, j]
            a = np.sum(c_1 * c_2)
            b = np.sum(c_1 * (1 - c_2))
            c = np.sum((1 - c_1) * c_2)
            d = np.sum((1 - c_1) * (1 - c_2))
            if (a * d + b * c) == 0:
                Q[i, j] = 1
            else:
                Q[i, j] = (a * d - b * c) / (a * d + b * c)
            Q[j, i] = Q[i, j]
    Q_m = 0.0
    for i in range(S - 1):
        for j in range(i + 1, S):
            Q_m += 2 / (S * (S - 1)) * Q[i, j]
    return Q_m


def diversity(clf, pred_data=None, real_targ=None):
    """
    Implemented directly from MATLAB, not pythonic.

    :param clf: Predictor.
    :param pred_data: Not used.
    :param real_targ: Not used.
    :return:
    """
    ensemble_size = getattr(clf, 'size', None)
    div = 0.0
    count = 1
    if not ensemble_size is None:
        for s in range(ensemble_size):
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


metric_dict = {
    'accuracy': accuracy,
    'rmse': rmse,
    'diversity': diversity,
}


def loss(clf, pred_data, real_targ, metric='accuracy'):
    """
    Inverse of the accuracy. It is used for cross validation.

    :param clf: classifier with predict method.
    :param pred_data: array of the targets
        according to the classifier.
    :param numpy.array real_targ: array of the real targets.
    :param bool real_targ: array of the real targets.
    :param str metric: metric to use
    :return:
    """
    metric_fun = metric_dict[metric]
    metric_value = metric_fun(clf=clf, pred_data=pred_data, real_targ=real_targ)
    if metric is 'rmse':
        return metric_value
    else:
        return 1.0 - metric_value
