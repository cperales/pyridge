from .cv_method import CVMethod
import numpy as np
from scipy.special import expit

neuron_fun_dict = {'sin': np.sin,
                   'hard': lambda x: np.array(x > 0.0, dtype=float),
                   # 'sigmoid': lambda x: 1.0/(1.0 + np.exp(-x))
                   'sigmoid': expit}


class NeuralMethod(CVMethod):
    # Cross validated parameters
    neuron_fun = None
    hidden_neurons = 0
    lambda_nc = 0

    # Neural network features
    input_weight = 0
    bias_vector = 0
    output_weight = 0
    t = 2  # At least, 2 labels are classified

    def set_conf(self, method_conf):
        # Neuron function
        self.neuron_fun = neuron_fun_dict[method_conf['neuronFun']]
        # Number of neurons in the hidden layer
        self.grid_param['hidden_neurons'] = np.array(method_conf['hiddenNeurons'])
        # Regularization
        self.grid_param['C'] = np.array(method_conf['C']) if 'C' in method_conf \
            else np.array([0], dtype=np.float)
        # Ensemble
        self.ensemble_size = np.array(method_conf['ensembleSize']) if 'ensembleSize' in method_conf \
            else np.array([1], dtype=np.float)
        # Negative correlation
        self.grid_param['lambda_nc'] = np.array(method_conf['lambda']) if 'lambda' in method_conf \
            else np.array([0], dtype=np.float)
        # Diversity
        self.grid_param['D'] = np.array(method_conf['D']) if 'D' in method_conf \
            else np.array([0], dtype=np.float)
