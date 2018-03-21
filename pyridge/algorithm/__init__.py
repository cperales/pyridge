from .neural import NRidge
from .adaboost import AdaBoostNRidge
from .kernel import KRidge

algorithm_dict = {'NRidge': NRidge,
                  'AdaBoostNRidge': AdaBoostNRidge,
                  'KRidge': KRidge}
