import logging
import warnings
from .algorithm import KRidge, AdaBoostNRidge, NRidge, algorithm_dict

__all__ = ['NRidge',
           'KRidge',
           'AdaBoostNRidge',
           'algorithm_dict']

warnings.simplefilter('ignore')

logger_pyelm = logging.Logger('PyRidge')
logger_pyelm.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logger_pyelm.debug('Logger instanced')
