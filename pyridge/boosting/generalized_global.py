from ..util.activation import activation_dict
from ..neural.elm import ELM
from ..util import solver
import numpy as np
import logging

logger = logging.getLogger('pyridge')


class GeneralizedGlobalBRELM(ELM):
    """
    Generalized Global Boosting Ridge ELM.
    """
    __name__ = 'Generalized Global Boosting Ridge ELM'
    input_weight: np.array
    bias_vector: np.array
    # Ensemble
    size: int

    def fit(self, train_data, train_target, parameter: dict):
        """
        Use some train (data and target) and parameter to
        fit the classifier and construct the rules.

        :param numpy.array train_data: data with features.
        :param numpy.array train_target: targets in j codification.
        :param dict parameter:
        """
        self.instance_param_(train_data=train_data,
                             train_target=train_target,
                             parameter=parameter)
        self.train()

    def train(self):
        """
        Train after instancing parameters and data.

        :return:
        """
        self.get_weight_bias_()
        H = self.get_h_matrix(data=self.train_data)

        A = self.get_A(H=H)
        right = self.get_right(H=H)

        self.output_weight = solver(A, right)

    def get_weight_bias_(self):
        """
        Return input weight bigger than usual.
        """
        self.neuron_fun = activation_dict[self.activation]
        self.input_weight = np.empty((self.size, self.hidden_neurons, self.dim))
        self.bias_vector = np.empty((self.size, self.hidden_neurons, 1))
        for s in range(self.size):
            self.input_weight[s] = np.random.rand(self.hidden_neurons,
                                                  self.dim) * 2.0 - 1.0
            self.bias_vector[s] = np.random.rand(self.hidden_neurons,
                                                 1)

    def get_h_s_matrix(self, data, s):
        """

        :param data:
        :param s:
        :return:
        """
        temp_h_s_matrix = (np.dot(self.input_weight[s], data.T) +
                           self.bias_vector[s]).T
        h_s_matrix = self.neuron_fun(temp_h_s_matrix)
        return h_s_matrix

    def get_h_matrix(self, data):
        """

        :param data:
        :return:
        """
        h_matrix = np.array([self.get_h_s_matrix(data=data, s=s)
                             for s in range(self.size)], dtype=np.float)
        return h_matrix

    def get_right(self, H):
        """

        :param H:
        :return:
        """
        right = np.concatenate([np.dot(H[s].T, self.Y)
                                for s in range(self.size)],
                               axis=0)
        return right

    def get_A(self, H):
        """
        Get A matrix by multiplicating H.

        :param H:.
        :return:
        """
        A = self.reg * np.eye(self.hidden_neurons * self.size)
        for i in range(self.size):
            for j in range(i, self.size):
                # Indexes
                k = i * self.hidden_neurons
                k_1 = (i + 1) * self.hidden_neurons
                t = j * self.hidden_neurons
                t_1 = (j + 1) * self.hidden_neurons
                # Slicing submatrix
                H_i_H_j = np.dot(H[i].T, H[j])
                if i == j:
                    A[k:k_1, t:t_1] += H_i_H_j
                else:
                    A[k:k_1, t:t_1] = H_i_H_j
                    A[t:t_1, k:k_1] = H_i_H_j.T
        return A

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: indicator.
        """
        indicator = np.dot(np.concatenate(self.get_h_matrix(data=test_data), axis=1),
                           self.output_weight)
        return indicator
