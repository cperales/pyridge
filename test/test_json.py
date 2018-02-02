# Tests
from test_NELM import test_newthyroid_json as test_NELM
from test_KELM import test_newthyroid_json as test_KELM
from test_AdaBoostNELM import test_newthyroid_json as test_AdaBoostNELM

# Other libraries
import logging

logger = logging.getLogger('PyELM')
logger.setLevel(logging.DEBUG)

logging.info('Starting tests...')
logging.info('Running Kernel Extreme Learning Machine test')
test_NELM()
logging.info('Running Neural Extreme Learning Machine test')
test_KELM()
logging.info('Running AdaBoost Neural Extreme Learning Machine test')
test_AdaBoostNELM()
logging.info('Tests have finished!')
