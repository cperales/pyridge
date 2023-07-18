from .elm import ELM
from .bagging import BaggingELM
from .bselm import BaggingStepwiseELM
from .adaboost import AdaBoostELM
from .adaboost_nc import AdaBoostNCELM
from .regularized import RegularizedEnsembleELM
from .diverse import DiverseELM
from .nn import NeuralNetwork
from .rnn import RandomNeuralNetwork
from .pca import PCAELM
from .lda import PCALDAELM
from .sobol import SobolELM
from .pl import ParallelLayerELM


neural_algorithm = {
    'ELM': ELM,
    'AdaBoostELM': AdaBoostELM,
    'BaggingELM': BaggingELM,
    'BaggingStepwiseELM': BaggingStepwiseELM,
    'AdaBoostNCELM': AdaBoostNCELM,
    'RegularizedEnsembleELM': RegularizedEnsembleELM,
    'REELM': RegularizedEnsembleELM,  # Another way of writing it
    'DiverseELM': DiverseELM,
    'NeuralNetwork': NeuralNetwork,
    'RandomNeuralNetwork': RandomNeuralNetwork,
    'PCAELM': PCAELM,
    'PCALDAELM': PCALDAELM,
    'SobolELM': SobolELM,
    'ParallelLayerELM': ParallelLayerELM,
}
