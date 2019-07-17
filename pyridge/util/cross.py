import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold, KFold
from pyridge.util.metric import loss
import logging

logger = logging.getLogger('pyridge')


def cross_validation(predictor,
                     train_data,
                     train_target,
                     hyperparameter,
                     metric='accuracy',
                     n_folds=5):
    """
    Cross validation training in order to find best parameter.

    :param predictor:
    :param train_data:
    :param train_target:
    :param dict hyperparameter:
    :param str metric:
    :param int n_folds:
    :return:
    """
    cv_param_names = list(hyperparameter.keys())
    list_comb = [hyperparameter[name] for name in cv_param_names]
    best_cv_criteria = np.inf
    if metric == 'accuracy':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True)

    for current_comb in itertools.product(*list_comb):
        loss_vector = list()
        param = {cv_param_names[i]: current_comb[i]
                 for i in range(len(cv_param_names))}
        logger.debug('Trying with parameter: %s', param)

        for train_index, test_index in kf.split(train_data, train_target):
            # TRAIN
            train_data_fold = train_data[train_index]
            train_target_fold = train_target[train_index]
            predictor.fit(train_data=train_data_fold,
                          train_target=train_target_fold,
                          parameter=param)

            # PREDICT
            test_data_fold = train_data[test_index]
            test_target_fold = train_target[test_index]

            l_value = loss(clf=predictor,
                           pred_data=test_data_fold,
                           real_targ=test_target_fold,
                           metric=metric)
            loss_vector.append(l_value)

        # loss_vector = np.array(loss_vector, dtype=np.float)
        current_cv_criteria = np.mean(loss_vector)

        logger.debug('With these parameter, test loss is %f',
                     current_cv_criteria)

        if current_cv_criteria < best_cv_criteria:
            best_param = param
            best_cv_criteria = current_cv_criteria

    logger.debug('Loss: %f; Cross validated parameter: %s',
                 best_cv_criteria, best_param)
    return best_param


def train_predictor(predictor,
                    train_data,
                    train_target,
                    hyperparameter,
                    metric='accuracy',
                    n_folds=5):
    """
    Cross validation training in order to find best parameter.

    :param predictor:
    :param train_data:
    :param train_target:
    :param dict hyperparameter:
    :param str metric:
    :param int n_folds:
    :return:
    """
    best_param = cross_validation(predictor=predictor,
                                  train_data=train_data,
                                  train_target=train_target,
                                  hyperparameter=hyperparameter,
                                  metric=metric,
                                  n_folds=n_folds)
    predictor.fit(train_data=train_data,
                  train_target=train_target,
                  parameter=best_param)
    return predictor
