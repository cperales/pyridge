from pyridge.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyridge')
logger.setLevel('INFO')


def test_adaboost_nc():
    hyperparameter = {'activation': ['sigmoid'],
                      'reg': [10 ** i for i in range(-2, 3)],
                      'hidden_neurons': [1000],
                      'lambda_': [0.5, 1.0, 5.0],
                      'size': [5]}
    metrics = ['accuracy', 'diversity', 'rmse']
    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm='AdaBoostNCELM',
                   metric=metrics,
                   hyperparameter=hyperparameter)
