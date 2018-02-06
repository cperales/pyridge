import logging
logger_pyelm = logging.Logger('PyELM')
logger_pyelm.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logger_pyelm.debug('Logger instanced')

import warnings
warnings.simplefilter('ignore')

from .algorithm import *
