from .cv_method import CVMethod


neuron_fun_dict = {'sin': np.sin,
                   'hard': lambda x: np.array(x > 0.0, dtype=float),
                   'sigmoid': lambda x: 1.0/(1.0 + np.exp(-x))}


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
        self.grid_param['hidden_neurons'] = method_conf['hidden_neurons']
        # Regularization
        self.grid_param['C'] = method_conf['C'] if 'C' in method_conf else 0
        # Ensemble
        self.ensemble_size = method_conf['ensembleSize'] if 'ensembleSize' in method_conf else 1
        # Negative correlation
        self.grid_param['lambda_nc'] = method_conf['lambda'] if 'lambda' in method_conf else 0
        # Diversity
        self.grid_param['D'] = method_conf['D'] if 'D' in method_conf else 0
