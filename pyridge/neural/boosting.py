from .neural import NeuralRidge
import numpy as np


class BoostingRidgeNRidge(NeuralRidge):
    """
    Boosting Ridge ensemble applied to Neural Ridge.
    """
    __name__ = 'Boosting Ridge Neural Ridge'
    size: int = 2
    Y_mu = None
    alpha = None

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
        self.alpha = np.ones(self.size)
        self.get_weight_bias_()
        h_matrix = self.get_h_matrix_(data=train_data)
        self.Y_mu = self.Y.astype(np.float64)

        self.output_weight = np.zeros((self.size, self.hidden_neurons, self.Y.shape[1]))
        for s in range(self.size):
            self.output_weight[s] = self.fit_step(h_matrix, s)
        self.output_weight[np.isnan(self.output_weight)] = 0.0

    def fit_step(self, h_matrix, s):
        """
        Each  step of the fit process.

        :param h_matrix:
        :param int s:
        :return :
        """
        # Beta is calculated
        izq = np.eye(h_matrix.shape[1]) + \
              self.reg * np.dot(h_matrix.T, h_matrix)
        der = np.dot(h_matrix.T, self.Y_mu)
        output_weight_s = np.linalg.solve(a=izq, b=der)

        # Y_mu updated
        y_pred = np.dot(h_matrix, output_weight_s)
        self.Y_mu -= y_pred

        return output_weight_s

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        h_matrix = self.get_h_matrix_(data=test_data)
        indicator = np.sum([self.alpha[s] * np.dot(h_matrix,
                                                   self.output_weight[s])
                            for s in range(len(self.output_weight))], axis=0)
        return indicator
