import json
import logging
from test_standard import run_test
import logging

logger = logging.getLogger('PyELM')
logger.setLevel(logging.DEBUG)


def test_newthyroid():
    # Reading JSON
    with open('config/KELM_newthyroid.json', 'r') as cfg:
        config_options = json.load(cfg)

    logger.info('Running test {}'.format(config_options['Data']['folder']))
    run_test(config_options)


def multi_test():
    # Reading JSON
    with open('config/KELM_multiprueba.json', 'r') as cfg:
        config_options = json.load(cfg)

    # Training data and target
    if isinstance(config_options, list):
        for config_option in config_options:
            logger.info('Running test {}'.format(config_option['Data']['folder']))
            run_test(config_option)
    else:
        run_test(config_options)


if __name__ == '__main__':
    test_newthyroid()

