from pyridge.util.activation import activation_dict
from .predictor import Predictor
import numpy as np
import logging

logger = logging.getLogger('pyridge')
_TOL_ = 10 ** -8


class TensorELM(Predictor):
    """
    Tensor ELM
    """
    __name__ = 'Tensor ELM'
    # Neural network
    hidden_neurons: int
    activation: str
    neuron_fun = None
    input_weight = None
    bias_vector = None
    # Ensemble
    size: int
    iter_: int
    H = None
    f = None
    beta = None

    def get_weight_bias_(self):
        """
        """
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight = np.random.rand(self.size,
                                      self.hidden_neurons,
                                      self.dim) * 2.0 - 1.0
        self.bias_vector = np.random.rand(self.size,
                                     self.hidden_neurons,
                                     1)

    def get_h_matrix(self, data, s):
        """

        :param data:
        :param int s: particular base learner of the ensemble.
        :return:
        """
        temp_h_matrix = (np.dot(self.input_weight[s], data.T) +
                         self.bias_vector[s]).T
        h_matrix = self.neuron_fun(temp_h_matrix)
        return h_matrix

    def get_f(self):
        """
        Get the j column of the indicator of training
        data, with beta normalized.

        :return: predicted labels.
        """
        return np.mean([np.dot(self.H[s],
                               self.output_weight[s])
                        for s in range(self.size)], axis=0)

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = np.mean([np.dot(self.get_h_matrix(data=test_data, s=s),
                                    self.output_weight[s])
                             for s in range(self.size)], axis=0)
        return indicator
