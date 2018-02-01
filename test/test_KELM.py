import json
import os
from time import perf_counter

import pandas as pd
from sklearn import preprocessing
import logging

from pyelm.algorithm import *
from pyelm.utils import save_classifier
from pyelm.utils.cross_val import *
from pyelm.utils.metric import accuracy
from pyelm.utils.target_encode import j_encode

logger_pyelm = logging.getLogger('PyELM')
logger_pyelm.setLevel(logging.DEBUG)


def test_newthyroid():
    # Reading JSON
    with open('config/KELM_newthyroid.json', 'r') as cfg:
        config_options = json.load(cfg)

    logger.info('Running test {}'.format(config_options['Data']['folder']))
    run_test(config_options)


def multi_test():
    # Reading JSON
    with open('config/KELM_multiprueba.json', 'r') as cfg:
        config_options = json.load(cfg)

    # Training data and target
    if isinstance(config_options, list):
        for config_option in config_options:
            logger.info('Running test {}'.format(config_option['Data']['folder']))
            run_test(config_option)
    else:
        run_test(config_options)


def run_test(config_test):
    training_file_name = os.path.join(config_test['Data']['folder'],
                                      config_test['Data']['dataset'][0][0])
    training_file = pd.read_csv(training_file_name,
                                sep='\s+',
                                header=None)
    training_file_matrix = training_file.as_matrix()
    training_file_matrix_t = training_file_matrix.transpose()
    training_target = training_file_matrix_t[-1].transpose()
    training_data = training_file_matrix_t[:-1].transpose()

    # Testing data and target
    testing_file_name = os.path.join(config_test['Data']['folder'],
                                     config_test['Data']['dataset'][0][1])
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

    clf.set_range_param(hyperparameters)
    training_J_target = j_encode(training_target)
    n_targ = training_J_target.shape[1]
    testing_j_target = j_encode(testing_target, n_targ=n_targ)

    train_dict = {'data': training_data, 'target': training_J_target}

    # # Fitting classifier
    # Profiling
    from cProfile import Profile
    prof = Profile()
    prof.enable()
    time_1 = perf_counter()

    n_run = 1
    acc = 0
    for i in range(n_run):
        cross_validation(classifier=clf, train=train_dict)
        predicted_labels = clf.predict(test_data=testing_data)
        acc += accuracy(predicted_targets=predicted_labels,
                        real_targets=testing_j_target)
    acc = acc / n_run

    # Saving classifier
    save_classifier(clf, 'KELM_newthyroid.clf')

    # Profiling
    time_2 = perf_counter()
    prof.disable()  # don't profile the generation of stats

    try:
        prof.dump_stats('profile/KELM_newthyroid.prof')
    except FileNotFoundError:  # There is no 'profile' folder
        pass

    logger_pyelm.debug('{} seconds elapsed'.format(time_2 - time_1))

    logger_pyelm.info('Average accuracy in {} iterations, \
        algorithm {} and dataset {} is {}'.format(n_run,
                                                  config_test['Algorithm']['name'],
                                                  config_test['Data']['dataset'][0][0],
                                                  acc))


if __name__ == '__main__':
    test_newthyroid()
