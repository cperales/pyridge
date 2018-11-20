from pyridge.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyridge')
logger.setLevel('INFO')


if __name__ == '__main__':
    hyperparameter = {'activation': ['sigmoid'],
                      'reg': [10 ** i for i in range(-1, 2)],
                      'div': [10 ** i for i in range(-1, 2)],
                      'hidden_neurons': [100],
                      'size': [5]}
    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm='DiverseELM',
                   hyperparameter=hyperparameter)
