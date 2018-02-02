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


def test_newthyroid():
    """
    Simple test with a UCI database.
    """
    # Data
    folder = 'data/newthyroid'
    train_dataset = 'train_newthyroid.0'
    test_dataset = 'test_newthyroid.0'

    train_file_name = os.path.join(folder,
                                      train_dataset)
    train_file = pd.read_csv(train_file_name,
                                sep='\s+',
                                header=None)
    train_file_matrix = train_file.as_matrix()
    train_file_matrix_t = train_file_matrix.transpose()
    train_target = train_file_matrix_t[-1].transpose()
    train_data = train_file_matrix_t[:-1].transpose()

    # Testing data and target
    test_file_name = os.path.join(folder,
                                     test_dataset)
    test_file = pd.read_csv(test_file_name,
                               sep='\s+',
                               header=None)
    test_file_matrix = test_file.as_matrix()
    test_file_matrix_t = test_file_matrix.transpose()
    test_target = test_file_matrix_t[-1].transpose()
    test_data = test_file_matrix_t[:-1].transpose()

    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)

    train_j_target = j_encode(train_target)
    n_targ = train_j_target.shape[1]
    test_j_target = j_encode(test_target, n_targ=n_targ)

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
    clf.set_cv_range(hyperparameters)
    cross_validation(classifier=clf, train_data=train_data, train_target=train_j_target)
    pred_targ = clf.predict(test_data=test_data)
    acc = metric(predicted_targets=pred_targ,
                 real_targets=test_j_target)

    logger.info('Accuracy for algorithm KELM and dataset newthyroid.0,'
                ' is {}'.format(acc))


if __name__ == '__main__':
    test_newthyroid()

