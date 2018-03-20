from .nelm import NRidge
from .adaboost import AdaBoostNRidge
from .kelm import KRidge

algorithm_dict = {'NRidge': NRidge,
                  'AdaBoostNRidge': AdaBoostNRidge,
                  'KRidge': KRidge}
