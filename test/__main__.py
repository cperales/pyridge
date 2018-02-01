# Tests
from test_NELM import test_newthyroid as test_NELM
from test_KELM import test_newthyroid as test_KELM
from test_AdaBoostNELM import test_newthyroid as test_AdaBoostNELM

# Other libraries
from multiprocessing import Process
import logging

logger_pyelm = logging.getLogger('PyELM')
logger_pyelm.setLevel(logging.INFO)

logging.info('Starting tests...')
logging.info('Running Kernel Extreme Learning Machine test')
p_k = Process(target=test_KELM)
p_k.start()
logging.info('Running Neural Extreme Learning Machine test')
p_n = Process(target=test_NELM)
p_n.start()
logging.info('Running AdaBoost Neural Extreme Learning Machine test')
p_an = Process(target=test_AdaBoostNELM)
p_an.start()

logging.info('Finishing tests...')
p_k.join()
p_n.join()
p_an.join()

logging.info('Tests have finished!')
