import numpy as np
from .predictor import Predictor
from ..util.activation import activation_dict


class NeuralMethod(Predictor):
    __name__ = 'Neural network'
    hidden_neurons: int
    activation: str
    neuron_fun = None
    input_weight = None
    bias_vector = None

    def get_weight_bias_(self):
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight = np.random.rand(self.hidden_neurons,
                                           self.dim) * 2.0 - 1.0
        self.bias_vector = np.random.rand(self.hidden_neurons,
                                          1)

    def get_h_matrix(self, data):
        """

        :param data:
        :return:
        """
        temp_h_matrix = (np.dot(self.input_weight, data.T) +
                         self.bias_vector).T
        h_matrix = self.neuron_fun(temp_h_matrix)
        return h_matrix
