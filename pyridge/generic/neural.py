import numpy as np
from scipy.special import expit

from pyridge.generic.classifier import Classifier

neuron_fun_dict = {'sin': np.sin,
                   'hard': lambda x: np.array(x > 0.0, dtype=float),
                   # 'sigmoid': lambda x: 1.0/(1.0 + np.exp(-x))
                   'sigmoid': expit}


class NeuralMethod(Classifier):
    __name__ = 'Neural network'
    # Cross validated parameters
    neuron_fun = expit
    hidden_neurons = 0
    lambda_nc = 0
    C = 0
    ensemble_size = 1
    grid_param = {'C': C}

    # Neural network features
    input_weight = 0
    bias_vector = 0
    output_weight = 0
    t = 2  # At least, 2 labels are classified

    def set_cv_range(self, hyperparameters):
        # Neuron function
        self.neuron_fun = neuron_fun_dict[hyperparameters['neuronFun']]
        # Number of neurons in the hidden layer
        self.grid_param['hidden_neurons'] = \
            np.array(hyperparameters['hiddenNeurons'])
        # Regularization
        self.grid_param['C'] = np.array(hyperparameters['C']) if \
            'C' in hyperparameters \
            else np.array([0], dtype=np.float)
        # Ensemble
        self.ensemble_size = hyperparameters['ensembleSize'] if \
            'ensembleSize' \
            in hyperparameters else 1
        # Negative correlation
        self.grid_param['lambda_nc'] = np.array(hyperparameters['lambda']) \
            if 'lambda' in hyperparameters \
            else np.array([0], dtype=np.float)
        # Diversity
        self.grid_param['D'] = np.array(hyperparameters['D']) if \
            'D' in hyperparameters else np.array([0], dtype=np.float)
