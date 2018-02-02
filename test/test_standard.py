import os
from time import perf_counter

import pandas as pd
from sklearn import preprocessing

from pyelm.algorithm import *
from pyelm.utils import save_classifier
from pyelm.utils.cross_val import *
from pyelm.utils.metric import accuracy
from pyelm.utils.target_encode import j_encode
import logging

logger = logging.getLogger('PyELM')


def run_test(config_test):
    diff_time = 0
    acc = 0
    full_name_report = config_test['Algorithm']['name'] + '_' + config_test['Report']['report_name']
    n_folds = len(config_test['Data']['dataset'])
    for fold in range(n_folds):
        dataset_fold = config_test['Data']['dataset'][fold]
        logger.debug('Fold %s', dataset_fold)
        train_file_name = os.path.join(config_test['Data']['folder'],
                                          dataset_fold[0])
        train_file = pd.read_csv(train_file_name,
                                    sep='\s+',
                                    header=None)
        train_file_matrix = train_file.as_matrix()
        train_file_matrix_t = train_file_matrix.transpose()
        train_target = train_file_matrix_t[-1].transpose()
        train_data = train_file_matrix_t[:-1].transpose()

        # Testing data and target
        test_file_name = os.path.join(config_test['Data']['folder'],
                                         dataset_fold[1])
        test_file = pd.read_csv(test_file_name,
                                   sep='\s+',
                                   header=None)
        test_file_matrix = test_file.as_matrix()
        test_file_matrix_t = test_file_matrix.transpose()
        test_target = test_file_matrix_t[-1].transpose()
        test_data = test_file_matrix_t[:-1].transpose()

        train_data = preprocessing.scale(train_data)
        test_data = preprocessing.scale(test_data)

        # Reading parameters
        hyperparameters = config_test['Algorithm']['hyperparameters']

        # Instancing classifier
        # clf = algorithm_dict[config_test['Algorithm']['name']](hyperparameters)
        clf = algorithm_dict[config_test['Algorithm']['name']]()

        # cross_validation(clf, hyperparameters)

        clf.set_cv_range(hyperparameters)
        train_j_target = j_encode(train_target)
        n_targ = train_j_target.shape[1]
        test_j_target = j_encode(test_target, n_targ=n_targ)

        # # Fitting classifier
        # Profiling
        from cProfile import Profile
        prof = Profile()
        prof.enable()
        time_1 = perf_counter()

        n_run = 10
        acc_fold = 0
        for i in range(n_run):
            cross_validation(classifier=clf, train_data=train_data, train_target=train_j_target)
            predicted_labels = clf.predict(test_data=test_data)
            acc_fold += accuracy(predicted_targets=predicted_labels,
                                 real_targets=test_j_target)
        acc += acc_fold / (n_run * n_folds)

        # Saving classifier
        save_classifier(clf, full_name_report + '.clf')

        # Profiling
        time_2 = perf_counter()
        prof.disable()  # don't profile the generation of stats
        diff_time += (time_2 - time_1)

        try:
            prof.dump_stats('profile/' + full_name_report + '.prof')
        except FileNotFoundError:
            pass

    logger.debug('{} seconds elapsed'.format(diff_time))

    logger.info('Average accuracy in {} folds, with {} iterations per fold, '.format(n_folds, n_run,) +
                'algorithm {} and dataset {} is {}'.format(config_test['Algorithm']['name'],
                                                           config_test['Report']['report_name'],
                                                           acc))
