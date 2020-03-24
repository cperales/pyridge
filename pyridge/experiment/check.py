from pyridge.util import metric_dict, prepare_data
from pyridge.util.cross import get_best_param, product_dict
from pyridge import algorithm_dict
import numpy as np
import logging
import os
from time import perf_counter


logger = logging.getLogger('pyridge')


def check_algorithm(folder,
                    dataset,
                    algorithm,
                    metric_list,
                    hyperparameter,
                    metric_cross=None,
                    classification=True,
                    autoencoder=False):
    """
    Testing easily a predictor along all the folds.

    :param str folder:
    :param str dataset:
    :param str algorithm:
    :param list metric_list:
    :param str metric_cross:
    :param dict hyperparameter:
    :param bool classification: True if we want a classification;
        False if we are looking for a regression.
    :param bool autoencoder: True if we want autoencoder test;
        False if we are looking for a classic supervised test.
    :return: a dictionary, with the metrics.
    """
    logger.info('Starting algorithm %s', algorithm)
    if autoencoder is True:
        classification = False

    if metric_cross is None:
        if 'accuracy' in metric_list and classification is True:
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
        value_dict_fold = check_fold(folder_dataset=folder_dataset,
                                     train_dataset=train_dataset,
                                     test_dataset=test_dataset,
                                     algorithm=algorithm,
                                     metric_list=metric_list,
                                     metric_cross=metric_cross,
                                     hyperparameter=hyperparameter,
                                     classification=classification,
                                     autoencoder=autoencoder)
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


def check_fold(folder_dataset='data/iris',
               train_dataset='train_iris.0',
               test_dataset='test_iris.0',
               sep='\s+',
               algorithm='ELM',
               metric_list=['accuracy', 'rmse'],
               metric_cross='accuracy',
               hyperparameter=None,
               classification=True,
               autoencoder=False):
    """
    Generic test.

    :param str folder_dataset:
    :param str train_dataset:
    :param str test_dataset:
    :param str sep:
    :param str algorithm:
    :param list metric_list:
    :param dict hyperparameter:
    :param str metric_cross:
    :param bool classification: True if we want a classification;
        False if we are looking for a regression.
    :param bool autoencoder: True if we want autoencoder test;
        False if we are looking for a classic supervised test.
    :return: a dictionary, with the metrics.
    """
    # Data
    train_data, train_target, data_scaler, target_scaler = \
        prepare_data(folder=folder_dataset,
                     dataset=train_dataset,
                     sep=sep,
                     classification=classification)
    test_data, test_target, _, _ = prepare_data(folder=folder_dataset,
                                                dataset=test_dataset,
                                                sep=sep,
                                                data_scaler=data_scaler,
                                                classification=classification,
                                                target_scaler=target_scaler)
    if autoencoder is True:
        train_target = train_data
        test_target = train_target

    # Algorithm
    start = perf_counter()
    clf = algorithm_dict[algorithm](classification)
    if hyperparameter is None:
        best_param = None
        end_cross = start
        clf.fit(train_data=train_data, train_target=train_target)
    else:
        best_param = get_best_param(predictor=clf,
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
                                    test_dataset,
                                    v))

    value_dict['test_time'] = test_time
    cross_time = end_cross - start
    value_dict['cross_time'] = cross_time
    train_time = end_train - end_cross
    value_dict['train_time'] = train_time
    value_dict['sum_time'] = cross_time + train_time + test_time

    return value_dict


def check_hyperparameter_sensitivity(folder,
                                     dataset,
                                     algorithm,
                                     metric,
                                     hyp_range_dict,
                                     fixed_hyp,
                                     repetitions: int = 5,
                                     classification: bool = True):
    """
    Function to check hyperparameter sensitivity.

    :param str folder:
    :param str dataset:
    :param str algorithm:
    :param dict hyp_range_dict:
    :param dict fixed_hyp:
    :param int repetitions:
    :param bool classification:
    :return: A dataframe with the value of the
        hyperparameters and the chosen metric.
    """
    # Load dataset
    folder_dataset = os.path.join(folder, dataset)
    k_datasets = [('train_' + dataset + '.' + str(i),
                   'test_' + dataset + '.' + str(i))
                  for i in range(10)]
    final_length = 1
    for parameter, value_list in hyp_range_dict.items():
        final_length = final_length * len(value_list)
    logger.debug('Hyperparameter sensitivity test is going '
                 'to check {} combinations'.format(final_length))
    # Prepare result dict
    result_dict = {metric: np.empty(final_length)}
    for parameter in list(hyp_range_dict.keys()):
        result_dict.update({parameter: np.empty(final_length)})
    count = 0
    logger.debug('Fixed parameters: %s', fixed_hyp)
    for hyp_combination in product_dict(**hyp_range_dict):
        logger.debug('Trying with parameter: %s, combination %i', hyp_combination, count)
        # Load algorithm
        fixed_hyp.update(hyp_combination)
        predictor = algorithm_dict[algorithm](classification)

        # K fold over repetitions
        metric_value = np.empty(10 * repetitions)
        i = 0
        for train_k_dataset, test_k_dataset in k_datasets:
            logger.debug('Fold %s', train_k_dataset.split('.')[-1])
            # Load data
            train_k_data, train_k_target, data_k_scaler, target_k_scaler = \
                prepare_data(folder=folder_dataset,
                             dataset=train_k_dataset,
                             classification=classification)
            test_k_data, test_k_target, _, _ = prepare_data(folder=folder_dataset,
                                                            dataset=test_k_dataset,
                                                            data_scaler=data_k_scaler,
                                                            classification=classification,
                                                            target_scaler=target_k_scaler)
            for r in range(repetitions):
                logger.debug('Repetition %i', r)
                # TRAIN
                predictor.fit(train_data=train_k_data,
                              train_target=train_k_target,
                              parameter=fixed_hyp)
                # TEST
                metric_value[i] = metric_dict[metric](clf=predictor,
                                                      pred_data=test_k_data,
                                                      real_targ=test_k_target)
                i += 1
        # Updating dict
        result_dict[metric][count] = metric_value.mean()
        for parameter, value in hyp_combination.items():
            result_dict[parameter][count] = value
        count += 1
    return result_dict


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
