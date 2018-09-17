import logging
import warnings
from .algorithm import KRidge, AdaBoostNRidge, NRidge, algorithm_dict

__all__ = ['NRidge',
           'KRidge',
           'AdaBoostNRidge',
           'algorithm_dict']

warnings.simplefilter('ignore')

logger_pyridge = logging.Logger('PyRidge')
logger_pyridge.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
logger_pyridge.debug('Logger instanced')
