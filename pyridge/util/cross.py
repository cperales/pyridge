import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold, KFold
from .metric import loss
import logging

logger = logging.getLogger('pyridge')


def product_dict(**kwargs):
    """
    Cartesian product of a dictionary.

    :param kwargs:
    :return:
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


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
    # cv_param_names = list(hyperparameter.keys())
    # list_comb = [hyperparameter[name] for name in cv_param_names]
    best_cv_criteria = np.inf
    if metric == 'accuracy':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True)

    for param in product_dict(**hyperparameter):
        logger.debug('Trying with parameter: %s', param)
        loss_vector = np.empty(n_folds)
        i = 0
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
            loss_vector[i] = l_value
            i += 1

        current_cv_criteria = loss_vector.mean()
        logger.debug('With these parameter, test loss is %f',
                     current_cv_criteria)

        if current_cv_criteria < best_cv_criteria:
            best_cv_criteria = current_cv_criteria
            best_param = param

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
    best_param = get_best_param(predictor=predictor,
                                train_data=train_data,
                                train_target=train_target,
                                hyperparameter=hyperparameter,
                                metric=metric,
                                n_folds=n_folds)
    predictor.fit(train_data=train_data,
                  train_target=train_target,
                  parameter=best_param)
    return predictor


def get_best_param(predictor,
                   train_data,
                   train_target,
                   hyperparameter,
                   metric,
                   n_folds=5):
    """

    :return:
    """
    cross = True
    for key, value in hyperparameter.items():
        if not isinstance(value, list):
            cross = False
            break

    if cross is True:
        best_param = cross_validation(predictor=predictor,
                                      train_data=train_data,
                                      train_target=train_target,
                                      hyperparameter=hyperparameter,
                                      metric=metric,
                                      n_folds=n_folds)
        logger.debug('Cross validated parameters for final training: %s', best_param)
    else:
        best_param = hyperparameter
        logger.debug('No cross validation, chosen parameters: %s', best_param)
    return best_param
