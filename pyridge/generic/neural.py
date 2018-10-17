import numpy as np
from scipy.special import expit
from .classifier import Classifier

activation_dict = {'sin': np.sin,
                   'hard': lambda x: np.array(x > 0.0, dtype=float),
                   'sigmoid': expit}


class NeuralMethod(Classifier):
    __name__ = 'Neural network'
    hidden_neurons: int = 2
    activation: str = 'sigmoid'
    neuron_fun = None
    input_weight = None
    bias_vector = None

    def get_weight_bias_(self):
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight = np.random.rand(self.hidden_neurons,
                                           self.dim) * 2.0 - 1.0
        self.bias_vector = np.random.rand(self.hidden_neurons,
                                          1)

    def get_h_matrix_(self, data):
        """

        :param data:
        :return:
        """
        n = data.shape[0]  # Number of instances of the data
        bias_matrix = np.resize(self.bias_vector.T,
                                (n, self.hidden_neurons)).T
        temp_h_matrix = np.dot(self.input_weight,
                               data.T) + bias_matrix  # d x n
        # Activation function
        h_matrix = self.neuron_fun(temp_h_matrix.T)  # n x d
        return h_matrix
