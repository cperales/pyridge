from pyelm.experiment.test import test_algorithm
import logging

logger = logging.getLogger('pyelm')
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
    hyperparameter_nelm = {'activation': ['sigmoid'],
                           'reg': [10 ** i for i in range(-1, 2)],
                           'hidden_neurons': [100]}
    hyperparameter_div = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'div': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [100],
                          'size': [5]}
    hyperparameter_kelm = {'kernel': ['rbf'],
                           'reg': [10 ** i for i in range(-1, 2)],
                           'gamma': [10 ** i for i in range(-1, 2)]}
    algorithms = [('AdaBoostNELM', hyperparameter_boost),
                  ('AdaBoostNCNELM', hyperparameter_nc),
                  ('BaggingNELM', hyperparameter_boost),
                  ('BoostingRidgeNELM', hyperparameter_boost),
                  ('DiverseNELM', hyperparameter_div),
                  ('NeuralELM', hyperparameter_nelm),
                  ('KernelELM', hyperparameter_kelm)]

    for alg_hyp in algorithms:
        logger.info('Starting algorithm %s', alg_hyp[0])
        test_algorithm(folder='data',
                       dataset='iris',
                       algorithm=alg_hyp[0],
                       hyperparameter=alg_hyp[1])
