from pyelm.algorithm import algorithm_dict
from pyelm.utils.preprocess import prepare_data

from test_kelm import test_newthyroid as test_kelm

import logging

logger = logging.getLogger('PyELM')
logger.setLevel(logging.DEBUG)

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


def test_nelm():
    """
    Simple test with a UCI database.
    """
    algorithm = algorithm_dict['NELM']
    clf = algorithm()
    clf.fit(train_data=train_data, train_target=train_j_target)
    pred_targ = clf.predict(test_data=test_data)


def test_adaboost():
    """
    Simple test with a UCI database.
    """
    algorithm = algorithm_dict['AdaBoostNELM']
    clf = algorithm()
    clf.ensemble_size = 3
    clf.fit(train_data=train_data, train_target=train_j_target)
    pred_targ = clf.predict(test_data=test_data)


if __name__ == '__main__':
    test_kelm()
    test_nelm()
    test_adaboost()
