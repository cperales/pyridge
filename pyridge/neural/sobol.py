from .elm import ELM
from pyscenarios.sobol import sobol
from ..util.activation import activation_dict
import logging
import numpy as np

logger = logging.getLogger('pyridge')


class SobolELM(ELM):
    """
    Sobol sequence used for nonlinear mapping ELM
    """
    def get_weight_bias_sobol(self):
        """
        Weight are obtained with Sobol Sequence
        :return:
        """
        # w_sobol = np.zeros((self.hidden_neurons, self.dim))
        # w_sobol[1:self.hidden_neurons] = sobol((self.hidden_neurons - 1,
        #                                         self.dim))
        return sobol((self.hidden_neurons,
                      self.dim))

    def get_weight_bias_(self):
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight = self.get_weight_bias_sobol()
        self.bias_vector = 0.0
