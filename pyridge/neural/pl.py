from .elm import ELM
from .sobol import SobolELM
from ..util.activation import activation_dict
import logging
import numpy as np

logger = logging.getLogger('pyridge')


class ParallelLayerELM(SobolELM):
    """
    Sobol sequence and uniform random used for nonlinear mapping ELM.
    """
    __name__ = 'Parallel Layer ELM'
    input_weight_sobol = None

    def get_weight_bias_uniform(self):
        self.input_weight = np.random.rand(self.hidden_neurons,
                                           self.dim) * 2.0 - 1.0
        self.bias_vector = np.random.rand(self.hidden_neurons,
                                          1)

    def get_weight_bias_(self):
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight_sobol = self.get_weight_bias_sobol()
        self.get_weight_bias_uniform()

    def get_h_matrix(self, data):
        """

        :param data:
        :return:
        """
        temp_h_matrix = (np.dot(self.input_weight, data.T) +
                         self.bias_vector).T
        h_matrix = self.neuron_fun(temp_h_matrix)

        temp_h_sobol = np.dot(data, self.input_weight_sobol.T)
        h_sobol = self.neuron_fun(temp_h_sobol)
        return h_sobol * h_matrix
