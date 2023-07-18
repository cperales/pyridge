import logging
import warnings
from .neural import *
from .kernel import *
from .negcor import *
from .linear import *
from .boosting import *

warnings.simplefilter('ignore')

algorithm_dict = neural_algorithm.copy()
algorithm_dict.update(kernel_algorithm)
algorithm_dict.update(nc_algorithm)
algorithm_dict.update(linear_dict)
algorithm_dict.update(boosting_dict)