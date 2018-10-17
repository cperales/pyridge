from pyelm.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyelm')
logger.setLevel('INFO')


if __name__ == '__main__':
    hyperparameter = {'activation': ['sigmoid'],
                      'reg': [10 ** i for i in range(-3, 4)],
                      'hidden_neurons': [1000]}
    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm='NeuralELM',
                   hyperparameter=hyperparameter)
