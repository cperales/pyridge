from ..util import metric_dict, prepare_data
from ..util.cross import cross_validation
from pyridge import algorithm_dict
import numpy as np
import logging
import os
from time import perf_counter


logger = logging.getLogger('pyridge')
hyperparameter_nelm = {'activation': ['sigmoid'],
                       'reg': [10 ** i for i in range(-3, 4)],
                       'hidden_neurons': [1000]}


def test_algorithm(folder='data/',
                   dataset='iris',
                   algorithm='ELM',
                   metric_list=['accuracy', 'rmse'],
                   hyperparameter=hyperparameter_nelm,
                   classification=True):
    """
    Testing easily a predictor along all the folds.

    :param str folder:
    :param str dataset:
    :param str algorithm:
    :param list metric_list:
    :param dict hyperparameter:
    :param bool classification: True if we want a classification;
        False if we are looking for a linear.
    :return: a dictionary, with the metrics.
    """
    logger.info('Starting algorithm %s', algorithm)
    if 'accuracy' in metric_list:
        metric_cross = 'accuracy'
    else:
        metric_cross = 'rmse'

    # Data
    folder_dataset = os.path.join(folder, dataset)
    datasets = [('train_' + dataset + '.' + str(i),
                 'test_' + dataset + '.' + str(i))
                for i in range(10)]
    value_dict = {m: np.empty(10) for m in metric_list}
    # Time
    value_dict['cross_time'] = np.empty(10)
    value_dict['train_time'] = np.empty(10)
    value_dict['test_time'] = np.empty(10)
    value_dict['sum_time'] = np.empty(10)

    fold = 0
    start = perf_counter()
    for train_dataset, test_dataset in datasets:
        value_dict_fold = test_fold(folder_dataset=folder_dataset,
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    algorithm=algorithm,
                                    metric_list=metric_list,
                                    metric_cross=metric_cross,
                                    hyperparameter=hyperparameter,
                                    classification=classification)
        for m, v in value_dict_fold.items():
            value_dict[m][fold] = v
        fold += 1
    end = perf_counter()
    sum_time = value_dict['sum_time'].copy()

    for m in value_dict.keys():
        value_dict[m] = np.mean(value_dict[m])
        if 'time' not in m:
            logger.info('{} for algorithm {} and '
                           'dataset {} is {}'.format(m,
                                                     algorithm,
                                                     dataset,
                                                     value_dict[m]))
    value_dict['time_std'] = np.std(sum_time)

    value_dict['time'] = end - start
    logger.info('Elapsed time is %f', value_dict['time'])

    return value_dict


def test_fold(folder_dataset='data/iris',
              train_dataset='train_iris.0',
              test_dataset='test_iris.0',
              algorithm='ELM',
              metric_list=['accuracy', 'rmse'],
              metric_cross='accuracy',
              hyperparameter=hyperparameter_nelm,
              classification=True):
    """
    Generic test.

    :param str folder_dataset:
    :param str train_dataset:
    :param str test_dataset:
    :param str algorithm:
    :param list metric_list:
    :param dict hyperparameter:
    :param str metric_cross:
    :param bool classification: True if we want a classification;
        False if we are looking for a linear.
    :return: a dictionary, with the metrics.
    """
    # Data
    train_data, train_target, scaler = \
        prepare_data(folder=folder_dataset,
                     dataset=train_dataset)
    test_data, test_target, _ = prepare_data(folder=folder_dataset,
                                             dataset=test_dataset,
                                             scaler=scaler)

    # Algorithm
    start = perf_counter()
    clf = algorithm_dict[algorithm](classification)
    best_param = cross_validation(predictor=clf,
                                  train_data=train_data,
                                  train_target=train_target,
                                  hyperparameter=hyperparameter,
                                  metric=metric_cross)
    end_cross = perf_counter()
    clf = algorithm_dict[algorithm](classification)
    clf.fit(train_data=train_data,
            train_target=train_target,
            parameter=best_param)
    end_train = perf_counter()

    # Metric
    test_time = 0.0
    value_dict = dict()
    for m in metric_list:
        metric_fun = metric_dict[m]
        start_test = perf_counter()
        v = metric_fun(clf=clf,
                       pred_data=test_data,
                       real_targ=test_target)
        test_time += perf_counter() - start_test
        value_dict.update({m: v})
        logger.debug('{} in dataset {} '
                     'is {}'.format(m,
                                    train_dataset.split('_')[1],
                                    v))

    value_dict['test_time'] = test_time
    cross_time = end_cross - start
    value_dict['cross_time'] = cross_time
    train_time = end_train - end_cross
    value_dict['train_time'] = train_time
    value_dict['sum_time'] = cross_time + train_time + test_time

    return value_dict


def create_logger(level='INFO', name='pyridge'):
    """
    Return a logger with time format.

    :param str level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    :param str name: name of the logger.
    :return: logger instanced.
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger
