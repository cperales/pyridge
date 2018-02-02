import json
import logging
from pyelm import logger_pyelm

logger_pyelm = logging.getLogger('PyELM')
logger_pyelm.setLevel(logging.DEBUG)

from test_standard import run_test


def test_newthyroid():
    # Reading JSON
    with open('config/AdaBoostNELM_newthyroid.json', 'r') as cfg:
        config_options = json.load(cfg)

    logger_pyelm.info('Running test {}'.format(config_options['Data']['folder']))
    run_test(config_options)


def multi_test():
    # Reading JSON
    with open('config/AdaBoostNELM_multiprueba.json', 'r') as cfg:
        config_options = json.load(cfg)

    # Training data and target
    if isinstance(config_options, list):
        for config_option in config_options:
            logger_pyelm.info('Running test {}'.format(config_option['Data']['folder']))
            run_test(config_option)
    else:
        run_test(config_options)


if __name__ == '__main__':
    test_newthyroid()
