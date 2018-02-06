from pyelm.utils.preprocess import prepare_data
from pyelm.utils import accuracy
from pyelm.utils import cross_validation
from pyelm.algorithm import algorithm_dict
from test_json import run_test

import json
from sklearn.svm import SVC
import numpy as np
import logging

logger = logging.getLogger('PyELM')
logger.setLevel(logging.DEBUG)


class SklearnSVC(SVC):
    """
    (LAYER CLASS) The object sklearn.svm.SVC with an structure that allows to use
    cross validation from PyELM library, in order to test against PyELM algorithms.

    :param SVC:
    :return:
    """
    grid_param = {}

    def __call__(self, parameters=None):
        self.C = parameters['C']
        self.gamma = parameters['k']
        self.kernel = 'rbf'

    def fit(self, train_data, train_target):
        super(SklearnSVC, self).fit(X=train_data, y=train_target)

    def predict(self, test_data):
        return super(SklearnSVC, self).predict(X=test_data)

    def set_cv_range(self, hyperparameters={'C': 0, 'k': 1, 'kernelFun': 'rbf'}):
        # Regularization
        self.grid_param['C'] = np.array(hyperparameters['C']) if 'C' in hyperparameters \
            else np.array([0], dtype=np.float)
        self.grid_param['k'] = np.array(hyperparameters['k']) if 'k' in hyperparameters \
            else np.array([1], dtype=np.float)

    def save_clf_param(self):
        return {'C': self.C,
                'k': self.gamma}


algorithm_dict.update({'sklearnSVC': SklearnSVC})


def sklearn_test_cv():
    # Data
    folder = '../data/hepatitis'
    train_dataset = 'train_hepatitis.0'
    train_data, train_target = prepare_data(folder=folder,
                                            dataset=train_dataset,
                                            j_encoding=False)
    test_dataset = 'test_hepatitis.0'
    test_data, test_target = prepare_data(folder=folder,
                                          dataset=test_dataset,
                                          j_encoding=False)

    # SVC sklearn
    clf = SklearnSVC()
    clf.fit(train_data=train_data, train_target=train_target)
    pred_targ = clf.predict(test_data=test_data)
    acc = accuracy(pred_targ=pred_targ, real_targ=test_target, j_encoded=False)  # To test it works
    clf.grid_param = dict()
    # clf.set_cv_range = KernelMethod.set_cv_range

    # Hyperparameters
    C_range = [10 ** i for i in range(-2, 3)]
    k_range = [10 ** i for i in range(-2, 3)]
    kernel_fun = 'rbf'
    hyperparameters = {'kernelFun': kernel_fun,
                       'C': C_range,
                       'k': k_range}

    clf.set_cv_range(hyperparameters)
    cross_validation(classifier=clf, train_data=train_data, train_target=train_target)
    pred_targ = clf.predict(test_data=test_data)
    acc = accuracy(pred_targ=pred_targ, real_targ=test_target, j_encoded=False)  # To test it works


def sklearn_comparison():
    logger.info('SVC test')
    # SVC
    with open('config/SVC_hepatitis.json', 'r') as cfg:
        config_SVC_hepatitis = json.load(cfg)
    run_test(config_SVC_hepatitis, n_run=10, j_encoding=False)

    logger.info('\n')
    logger.info('KELM test')
    # KELM
    with open('config/KELM_hepatitis.json', 'r') as cfg:
        config_KELM_hepatitis = json.load(cfg)
    run_test(config_KELM_hepatitis, n_run=10)


def sklearn_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    from pyelm import NELM
    check_estimator(NELM)


if __name__ == '__main__':
    sklearn_test_cv()
    sklearn_comparison()
    # sklearn_check_estimator()
