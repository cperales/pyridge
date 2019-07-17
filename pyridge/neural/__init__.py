from .elm import ELM
from .bagging import BaggingELM
from .bselm import BaggingStepwiseELM
from .adaboost import AdaBoostELM
from .boosting import BoostingRidgeELM
from .adaboost_nc import AdaBoostNCELM
from .regularized import RegularizedELM
from .diverse import DiverseELM


neural_algorithm = {
    'ELM': ELM,
    'AdaBoostELM': AdaBoostELM,
    'BoostingRidgeELM': BoostingRidgeELM,
    'BaggingELM': BaggingELM,
    'BaggingStepwiseELM': BaggingStepwiseELM,
    'AdaBoostNCELM': AdaBoostNCELM,
    'RegularizedELM': RegularizedELM,
    'REELM': RegularizedELM,  # Another way of writing it
    'DiverseELM': DiverseELM,
                    }
