from .boosting_ridge import BoostingRidgeELM
import numpy as np
from ..util.activation import activation_dict
from ..util import solver


class GeneralizedBRELM(BoostingRidgeELM):
    """
    Generalized Boosting Ridge ELM.
    """
    __name__ = 'Generalized Boosting Ridge ELM'
    input_weight: np.array
    bias_vector: np.array

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

    def fit(self, train_data, train_target, parameter):
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

        self.get_weight_bias_()
        self.alpha = np.ones(self.size)
        H = self.get_h_matrix(data=self.train_data)

        y_mu = self.Y.astype(np.float64)
        self.output_weight = np.zeros((self.size, self.hidden_neurons, self.t))
        for s in range(self.size):
            izq = np.eye(self.hidden_neurons) * self.reg + np.dot(H[s].T, H[s])
            self.output_weight[s] = self.fit_step(izq=izq,
                                                  h_matrix=H[s],
                                                  y_mu=y_mu)
            # Y_mu update
            y_mu -= np.dot(H[s], self.output_weight[s])
        self.output_weight[np.isnan(self.output_weight)] = 0.0

    def fit_step(self, izq, h_matrix,  y_mu=None):
        """
        Each  step of the fit process.

        :param izq: for signature.
        :param numpy.array h_matrix:
        :param numpy.array y_mu:
        :return:
        """
        der = np.dot(h_matrix.T, y_mu)
        return solver(a=izq, b=der)

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: f(X) vector.
        """
        indicator = np.sum([np.dot(self.get_h_s_matrix(data=test_data, s=s),
                                   self.output_weight[s])
                            for s in range(self.size)], axis=0)
        return indicator
