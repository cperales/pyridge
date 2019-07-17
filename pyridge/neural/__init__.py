from .elm import ELM
from .bagging import BaggingELM
from .bselm import BaggingStepwiseELM
from .adaboost import AdaBoostELM
from .boosting import BoostingRidgeELM
from .adaboost_nc import AdaBoostNCELM
from .regularized import RegularizedEnsembleELM
from .diverse import DiverseELM


neural_algorithm = {
    'ELM': ELM,
    'AdaBoostELM': AdaBoostELM,
    'BoostingRidgeELM': BoostingRidgeELM,
    'BaggingELM': BaggingELM,
    'BaggingStepwiseELM': BaggingStepwiseELM,
    'AdaBoostNCELM': AdaBoostNCELM,
    'RegularizedEnsembleELM': RegularizedEnsembleELM,
    'REELM': RegularizedEnsembleELM,  # Another way of writing it
    'DiverseELM': DiverseELM,
                    }
