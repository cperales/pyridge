import numpy as np
import itertools
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import logging
from pyelm.clf_utility import loss

logger = logging.getLogger('PyELM')


def cross_validation(classifier, train, n_folds=5):
    """

    :param classifier:
    :param range_parameters:
    :param train:
    :param n_folds:
    :return:
    """
    cv_param_names = list(classifier.grid_param.keys())
    list_comb = [classifier.grid_param[name] for name
                 in cv_param_names]

    # # Cross validation
    # Init the CV criteria
    best_cv_criteria = np.inf
    kf = KFold(n_splits=n_folds, shuffle=True)

    for current_comb in itertools.product(*list_comb):
        L = []
        clf_list = []

        for train_index, test_index in kf.split(train['data']):
            param = {cv_param_names[i]: current_comb[i]
                     for i in range(len(cv_param_names))}

            train_fold = {'data': train['data'][train_index],
                          'target': train['target'][train_index]}
            classifier.fit(train=train_fold, parameters=param)

            test_fold = train['data'][test_index]
            pred = classifier.predict(test_data=test_fold)

            clf_param = classifier.save_clf_param()
            clf_list.append(clf_param)

            test_fold_target = train['target'][test_index]
            L.append(loss(real_targets=test_fold_target, predicted_targets=pred))

        # L = np.array(L, dtype=np.float)
        current_cv_criteria = np.mean(L)

        if current_cv_criteria < best_cv_criteria:
            position = L.index(min(L))
            best_clf_param = clf_list[position]
            best_cv_criteria = current_cv_criteria

    logger.debug('Best parameters for cross validations: %s', best_clf_param)
    classifier.fit(train=train, parameters=best_clf_param)
