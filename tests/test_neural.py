from pyridge.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyridge')
logger.setLevel('INFO')


def test_neural():
    hyperparameter = {'activation': ['sigmoid'],
                      'reg': [10 ** i for i in range(-3, 4)],
                      'hidden_neurons': [1000]}
    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm='ELM',
                   hyperparameter=hyperparameter)
