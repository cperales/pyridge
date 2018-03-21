from pyridge.utils import metric_dict
from pyridge.algorithm import algorithm_dict
from pyridge.utils import cross_validation
from pyridge.utils.preprocess import prepare_data

import logging

logger = logging.getLogger('PyRidge')
logger.setLevel(logging.DEBUG)


def test_newthyroid():
    """
    Simple test with a UCI database.
    """
    # Data
    folder = 'data/newthyroid'
    train_dataset = 'train_newthyroid.0'
    train_data, train_j_target = prepare_data(folder=folder,
                                              dataset=train_dataset)
    test_dataset = 'test_newthyroid.0'
    n_targ = train_j_target.shape[1]

    test_data, test_j_target = prepare_data(folder=folder,
                                            dataset=test_dataset,
                                            n_targ=n_targ)

    # Algorithm
    metric = metric_dict['accuracy']
    algorithm = algorithm_dict['AdaBoostNRidge']
    C_range = [10**i for i in range(-2, 3)]
    neuron_range = [10*i for i in range(1, 21)]
    neural_fun = 'sigmoid'
    ensemble_size = 3

    hyperparameters = {'neuronFun': neural_fun,
                       'C': C_range,
                       'hiddenNeurons': neuron_range,
                       'ensembleSize': ensemble_size}

    clf = algorithm()
    clf.set_cv_range(hyperparameters)
    cross_validation(classifier=clf, train_data=train_data, train_target=train_j_target)
    pred_targ = clf.predict(test_data=test_data)
    acc = metric(pred_targ=pred_targ,
                 real_targ=test_j_target)

    logger.info('Accuracy for algorithm AdaBoostNRidge and dataset newthyroid.0,'
                ' is {}'.format(acc))


if __name__ == '__main__':
    test_newthyroid()
