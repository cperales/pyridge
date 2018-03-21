import logging
import warnings
from .algorithm import KRidge, AdaBoostNRidge, NRidge, algorithm_dict

__all__ = ['NRidge',
           'KRidge',
           'AdaBoostNRidge',
           'algorithm_dict']

warnings.simplefilter('ignore')

logger_pyridge = logging.Logger('PyRidge')
logger_pyridge.setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logger_pyridge.debug('Logger instanced')
