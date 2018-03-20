import logging
import warnings
from .algorithm import KELM, AdaBoostNELM, NELM, algorithm_dict

__all__ = ['NELM',
           'KELM',
           'AdaBoostNELM',
           'algorithm_dict']

warnings.simplefilter('ignore')

logger_pyelm = logging.Logger('PyELM')
logger_pyelm.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logger_pyelm.debug('Logger instanced')
