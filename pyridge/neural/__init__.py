from .neural import ELM
from .bagging import BaggingELM
from .adaboost import AdaBoostELM
from .boosting import BoostingRidgeELM
from .adaboost_nc import AdaBoostNCELM
from .diverse import DiverseELM

neural_algorithm = {'ELM': ELM,
                    'AdaBoostELM': AdaBoostELM,
                    'BoostingRidgeELM': BoostingRidgeELM,
                    'BaggingELM': BaggingELM,
                    'AdaBoostNCELM': AdaBoostNCELM,
                    'DiverseELM': DiverseELM}
