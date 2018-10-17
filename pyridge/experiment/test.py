from pyridge.util import cross_validation, metric_dict, prepare_data
from pyridge import algorithm_dict
import numpy as np
import logging
import os

logger = logging.getLogger('pyridge')
hyperparameter_nelm = {'activation': ['sigmoid'],
                       'reg': [10 ** i for i in range(-3, 4)],
                       'hidden_neurons': [1000]}

def test_algorithm(folder='data/',
                   dataset='iris',
                   algorithm='NeuralELM',
                   metric=['accuracy'],
                   hyperparameter=hyperparameter_nelm):
    """
    Generic test.

    :param str folder:
    :param str dataset:
    :param str algorithm:
    :param list metric:
    :param dict hyperparameter:
    :return:
    """
    # Data
    folder = os.path.join(folder, dataset)
    datasets = [('train_' + dataset + '.' + str(i),
                 'test_' + dataset + '.' + str(i))
                for i in range(10)]
    value_dict = {}
    for m in metric:
        value_dict[m] = list()
    for train_dataset, test_dataset in datasets:
        train_data, train_target, scaler = \
            prepare_data(folder=folder,
                         dataset=train_dataset)
        test_data, test_target, _ = prepare_data(folder=folder,
                                              dataset=test_dataset,
                                              scaler=scaler)
        clf = algorithm_dict[algorithm]()
        cross_validation(classifier=clf,
                         train_data=train_data,
                         train_target=train_target,
                         hyperparameter=hyperparameter)
        for m in metric:
            metric_fun = metric_dict[m]
            v = metric_fun(clf=clf,
                           pred_data=test_data,
                           real_targ=test_target)
            value_dict[m].append(v)
            logger.debug('{} in dataset {} '
                        'is {}'.format(m,
                                       train_dataset.split('_')[1],
                                       v))
    for m in metric:
        logger.info('{} for algorithm {} and '
                    'dataset {} is {}'.format(m,
                                              clf.__name__,
                                              dataset,
                                              np.mean(value_dict[m])))