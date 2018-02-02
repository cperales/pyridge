from pyelm.utils import metric_dict
from pyelm.algorithm import algorithm_dict
from pyelm.utils import cross_validation
from pyelm.utils.preprocess import prepare_data

import logging

logger = logging.getLogger('PyELM')
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
    algorithm = algorithm_dict['AdaBoostNELM']
    C_range = [10**i for i in range(-2, 3)]
    neuron_range = [10*i for i in range(1, 21)]
    neural_fun = 'sigmoid'
    ensemble_size = 5

    hyperparameters = {'neuronFun': neural_fun,
                       'C': C_range,
                       'hiddenNeurons': neuron_range,
                       'ensembleSize': ensemble_size}

    clf = algorithm()
    clf.set_cv_range(hyperparameters)
    cross_validation(classifier=clf, X=train_data, y=train_j_target)
    pred_targ = clf.predict(X=test_data)
    acc = metric(predicted_targets=pred_targ,
                 real_targets=test_j_target)

    logger.info('Accuracy for algorithm AdaBoostNELM and dataset newthyroid.0,'
                ' is {}'.format(acc))


if __name__ == '__main__':
    test_newthyroid()
