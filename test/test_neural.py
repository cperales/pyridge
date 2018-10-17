from pyridge.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyridge')
logger.setLevel('INFO')


if __name__ == '__main__':
    hyperparameter = {'activation': ['sigmoid'],
                      'reg': [10 ** i for i in range(-3, 4)],
                      'hidden_neurons': [1000]}
    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm='NeuralRidge',
                   hyperparameter=hyperparameter)
