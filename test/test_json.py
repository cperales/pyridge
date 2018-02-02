from pyelm.algorithm import *
from pyelm.utils import save_classifier
from pyelm.utils.cross_val import *
from pyelm.utils.metric import accuracy
from pyelm.utils.preprocess import prepare_data

import os
from time import perf_counter
import pandas as pd
from sklearn import preprocessing
import json
import logging


def run_test(config_test, n_run=10):
    diff_time = 0
    acc = 0
    full_name_report = config_test['Algorithm']['name'] + '_' + config_test['Report']['report_name']
    n_folds = len(config_test['Data']['dataset'])
    folder = config_test['Data']['folder']
    for fold in range(n_folds):
        # Data
        dataset_fold = config_test['Data']['dataset'][fold]
        logger.debug('Fold %s', dataset_fold)
        train_dataset = dataset_fold[0]
        train_data, train_j_target = prepare_data(folder=folder,
                                                  dataset=train_dataset)
        n_targ = train_j_target.shape[1]
        test_dataset = dataset_fold[1]

        test_data, test_j_target = prepare_data(folder=folder,
                                                dataset=test_dataset,
                                                n_targ=n_targ)

        # Reading parameters
        hyperparameters = config_test['Algorithm']['hyperparameters']

        # Instancing classifier
        # clf = algorithm_dict[config_test['Algorithm']['name']](hyperparameters)
        clf = algorithm_dict[config_test['Algorithm']['name']]()

        # cross_validation(clf, hyperparameters)
        clf.set_cv_range(hyperparameters)

        # # Fitting classifier
        # Profiling
        from cProfile import Profile
        prof = Profile()
        prof.enable()
        time_1 = perf_counter()

        acc_fold = 0
        for i in range(n_run):
            cross_validation(classifier=clf, X=train_data, y=train_j_target)
            predicted_labels = clf.predict(X=test_data)
            acc_fold += accuracy(pred_targ=predicted_labels,
                                 real_targ=test_j_target)
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


if __name__ == '__main__':
    logger = logging.getLogger('PyELM')
    logger.setLevel(logging.INFO)

    # Reading JSONs
    with open('config/KELM_newthyroid.json', 'r') as cfg:
        config_KELM_newthyroid = json.load(cfg)

    with open('config/NELM_newthyroid.json', 'r') as cfg:
        config_NELM_newthyroid = json.load(cfg)

    with open('config/AdaBoostNELM_newthyroid.json', 'r') as cfg:
        config_AdaBoostNELM_newthyroid = json.load(cfg)

    # Run tests
    logging.info('Starting tests...')
    logging.info('Running Kernel Extreme Learning Machine test')
    run_test(config_KELM_newthyroid)
    logging.info('Running Neural Extreme Learning Machine test')
    run_test(config_NELM_newthyroid)
    logging.info('Running AdaBoost Neural Extreme Learning Machine test')
    run_test(config_AdaBoostNELM_newthyroid)
    logging.info('Tests have finished!')
