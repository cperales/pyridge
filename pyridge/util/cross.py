import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from pyridge.util.metric import loss
import logging

logger = logging.getLogger('pyridge')


def cross_validation(classifier,
                     train_data,
                     train_target,
                     hyperparameter,
                     n_folds=5):
    """
    Cross validation training in order to find best parameter.

    :param classifier:
    :param train_data:
    :param train_target:
    :param dict hyperparameter:
    :param int n_folds:
    :return:
    """
    cv_param_names = list(hyperparameter.keys())
    list_comb = [hyperparameter[name] for name in cv_param_names]
    best_cv_criteria = 1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for current_comb in itertools.product(*list_comb):
        loss_vector = list()
        train_loss_vector = list()
        param = {cv_param_names[i]: current_comb[i]
                 for i in range(len(cv_param_names))}
        logger.debug('Trying with parameter: %s', param)

        for train_index, test_index in skf.split(train_data, train_target):
            # TRAIN
            train_data_fold = train_data[train_index]
            train_target_fold = train_target[train_index]
            classifier.fit(train_data=train_data_fold,
                           train_target=train_target_fold,
                           parameter=param)

            pred = classifier.predict(test_data=train_data_fold)
            l_value = loss(real_targets=train_target_fold,
                           predicted_targets=pred)
            train_loss_vector.append(l_value)

            # PREDICT
            test_data_fold = train_data[test_index]
            test_target_fold = train_target[test_index]

            pred = classifier.predict(test_data=test_data_fold)
            l_value = loss(real_targets=test_target_fold,
                           predicted_targets=pred)
            loss_vector.append(l_value)

        # loss_vector = np.array(loss_vector, dtype=np.float)
        current_cv_criteria = np.mean(loss_vector)

        logger.debug('With these parameter, train loss is %f, test loss is %f',
                     np.mean(train_loss_vector),
                     current_cv_criteria)

        if current_cv_criteria < best_cv_criteria:
            best_param = param
            best_cv_criteria = current_cv_criteria

    logger.debug('Loss: %f; Cross validated parameter: %s',
                 best_cv_criteria, best_param)
    # Training all data
    classifier.fit(train_data=train_data, train_target=train_target, parameter=best_param)
