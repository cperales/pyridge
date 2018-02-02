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
        training_file_name = os.path.join(config_test['Data']['folder'],
                                          dataset_fold[0])
        training_file = pd.read_csv(training_file_name,
                                    sep='\s+',
                                    header=None)
        training_file_matrix = training_file.as_matrix()
        training_file_matrix_t = training_file_matrix.transpose()
        training_target = training_file_matrix_t[-1].transpose()
        training_data = training_file_matrix_t[:-1].transpose()

        # Testing data and target
        testing_file_name = os.path.join(config_test['Data']['folder'],
                                         dataset_fold[1])
        testing_file = pd.read_csv(testing_file_name,
                                   sep='\s+',
                                   header=None)
        testing_file_matrix = testing_file.as_matrix()
        testing_file_matrix_t = testing_file_matrix.transpose()
        testing_target = testing_file_matrix_t[-1].transpose()
        testing_data = testing_file_matrix_t[:-1].transpose()

        training_data = preprocessing.scale(training_data)
        testing_data = preprocessing.scale(testing_data)

        # Reading parameters
        hyperparameters = config_test['Algorithm']['hyperparameters']

        # Instancing classifier
        # clf = algorithm_dict[config_test['Algorithm']['name']](hyperparameters)
        clf = algorithm_dict[config_test['Algorithm']['name']]()

        # cross_validation(clf, hyperparameters)

        clf.set_cv_range(hyperparameters)
        training_j_target = j_encode(training_target)
        n_targ = training_j_target.shape[1]
        testing_j_target = j_encode(testing_target, n_targ=n_targ)

        train_dict = {'data': training_data, 'target': training_j_target}

        # # Fitting classifier
        # Profiling
        from cProfile import Profile
        prof = Profile()
        prof.enable()
        time_1 = perf_counter()

        n_run = 10
        acc_fold = 0
        for i in range(n_run):
            cross_validation(classifier=clf, train=train_dict)
            predicted_labels = clf.predict(test_data=testing_data)
            acc_fold += accuracy(predicted_targets=predicted_labels,
                                 real_targets=testing_j_target)
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
