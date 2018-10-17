from pyridge.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyridge')
logger.setLevel('INFO')


if __name__ == '__main__':
    hyperparameter = {'kernel': ['rbf'],
                      'reg': [10 ** i for i in range(-3, 4)],
                      'gamma': [10 ** i for i in range(-2, 2)]}
    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm='KernelRidge',
                   hyperparameter=hyperparameter)
