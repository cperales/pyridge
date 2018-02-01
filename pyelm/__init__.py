import logging
logger_pyelm = logging.Logger('PyELM')
logger_pyelm.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
logger_pyelm.debug('Logger instanced')

import warnings
warnings.simplefilter('ignore')
