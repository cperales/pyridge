from test_standard import run_test
from pyelm.utils import metric_dict
from pyelm.algorithm import algorithm_dict
from pyelm.utils.target_encode import j_encode
from pyelm.utils import cross_validation

import os
import json
import pandas as pd
from sklearn import preprocessing
import logging

logger = logging.getLogger('PyELM')
logger.setLevel(logging.DEBUG)


def test_newthyroid_json():
    # Reading JSON
    with open('config/KELM_newthyroid.json', 'r') as cfg:
        config_options = json.load(cfg)

    logger.info('Running test {}'.format(config_options['Data']['folder']))
    run_test(config_options)


def multi_test_json():
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


def test_newthyroid():
    # Data
    folder = 'data/newthyroid'
    train_dataset = 'train_newthyroid.0'
    test_dataset = 'test_newthyroid.0'

    training_file_name = os.path.join(folder,
                                      train_dataset)
    training_file = pd.read_csv(training_file_name,
                                sep='\s+',
                                header=None)
    training_file_matrix = training_file.as_matrix()
    training_file_matrix_t = training_file_matrix.transpose()
    training_target = training_file_matrix_t[-1].transpose()
    training_data = training_file_matrix_t[:-1].transpose()

    # Testing data and target
    testing_file_name = os.path.join(folder,
                                     test_dataset)
    testing_file = pd.read_csv(testing_file_name,
                               sep='\s+',
                               header=None)
    testing_file_matrix = testing_file.as_matrix()
    testing_file_matrix_t = testing_file_matrix.transpose()
    testing_target = testing_file_matrix_t[-1].transpose()
    testing_data = testing_file_matrix_t[:-1].transpose()

    training_data = preprocessing.scale(training_data)
    testing_data = preprocessing.scale(testing_data)

    training_j_target = j_encode(training_target)
    n_targ = training_j_target.shape[1]
    testing_j_target = j_encode(testing_target, n_targ=n_targ)

    train_dict = {'data': training_data, 'target': training_j_target}

    # Algorithm
    metric = metric_dict['accuracy']
    algorithm = algorithm_dict['KELM']
    C_range = [10**i for i in range(-2, 3)]
    k_range = [10**i for i in range(-2, 3)]
    kernel_fun = 'rbf'

    hyperparameters = {'kernelFun': kernel_fun,
                       'C': C_range,
                       'k': k_range}

    clf = algorithm()
    clf.set_range_param(hyperparameters)
    cross_validation(classifier=clf, train=train_dict)
    pred_targ = clf.predict(test_data=testing_data)
    acc = metric(predicted_targets=pred_targ,
                 real_targets=testing_j_target)

    logger.info('Accuracy for algorithm KELM and dataset newthyroid.0,'
                ' is {}'.format(acc))


if __name__ == '__main__':
    test_newthyroid()

