import logging
import warnings
from .neural import *
from .kernel import *

warnings.simplefilter('ignore')

logger = logging.Logger('pyridge')
logging.debug('Logger instanced')

algorithm_dict = neural_algorithm.copy()
algorithm_dict.update(kernel_algorithm)
