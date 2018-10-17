from pyridge.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyridge')
logger.setLevel('INFO')


if __name__ == '__main__':
    hyperparameter_nc = {'activation': ['sigmoid'],
                         'reg': [10 ** i for i in range(-1, 2)],
                         'hidden_neurons': [100],
                         'lambda_': [0.5, 1.0, 5.0],
                         'size': [5]}
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [100],
                            'size': [5]}
    hyperparameter_neural = {'activation': ['sigmoid'],
                             'reg': [10 ** i for i in range(-1, 2)],
                             'hidden_neurons': [100]}
    hyperparameter_div = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'div': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [100],
                          'size': [5]}
    hyperparameter_kernel = {'kernel': ['rbf', 'linear'],
                             'reg': [10 ** i for i in range(-1, 2)],
                             'gamma': [10 ** i for i in range(-1, 2)]}
    algorithms = [('AdaBoostNRidge', hyperparameter_boost),
                  ('AdaBoostNCNRidge', hyperparameter_nc),
                  ('BaggingNRidge', hyperparameter_boost),
                  ('BoostingRidgeNRidge', hyperparameter_boost),
                  ('DiverseNRidge', hyperparameter_div),
                  ('NeuralRidge', hyperparameter_neural),
                  ('KernelRidge', hyperparameter_kernel)]

    for alg_hyp in algorithms:
        logger.info('Starting algorithm %s', alg_hyp[0])
        test_algorithm(folder='data',
                       dataset='iris',
                       algorithm=alg_hyp[0],
                       hyperparameter=alg_hyp[1],
                       metric=['accuracy', 'rmse'])
