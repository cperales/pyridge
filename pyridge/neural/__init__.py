from .neural import NeuralRidge
from .bagging import BaggingNRidge
from .adaboost import AdaBoostNRidge
from .boosting import BoostingRidgeNRidge
from .adaboost_nc import AdaBoostNCNRidge
from .diverse import DiverseNRidge

neural_algorithm = {'NeuralRidge': NeuralRidge,
                    'AdaBoostNRidge': AdaBoostNRidge,
                    'BoostingRidgeNRidge': BoostingRidgeNRidge,
                    'BaggingNRidge': BaggingNRidge,
                    'AdaBoostNCNRidge': AdaBoostNCNRidge,
                    'DiverseNRidge': DiverseNRidge}
