from pyridge.experiment.check import *
import logging

logger = logging.getLogger('testing')


def test_create_logger():
    create_logger('DEBUG', 'testing')
    logger.debug('Logger is created')


def test_hyp_sensitivity():
    result_dict = check_hyperparameter_sensitivity(folder='data',
                                                   dataset='iris',
                                                   algorithm='ELM',
                                                   metric='accuracy',
                                                   hyp_range_dict={'reg': [10**i for i in range(-2, 3)],
                                                                   'hidden_neurons': [10*i for i in range(1, 6)]},
                                                   fixed_hyp={'activation': 'sigmoid'},
                                                   repetitions=1,
                                                   classification=True)


if __name__ == '__main__':
    test_hyp_sensitivity()
