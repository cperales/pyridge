from ..neural.elm import ELM
import numpy as np
from ..util import solver


class BoostingRidgeELM(ELM):
    """
    Boosting Ridge ensemble applied to Neural ELM.
    """
    __name__ = 'Boosting Ridge Neural ELM'
    size: int
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
        h_matrix = self.instance_weight(train_data=train_data,
                                        train_target=train_target,
                                        parameter=parameter)
        self.alpha = np.ones(self.size)

        y_mu = self.Y.copy()
        self.output_weight = np.zeros((self.size, self.hidden_neurons, self.t))
        for s in range(self.size):
            self.output_weight[s] = self.fit_step(h_matrix=h_matrix,
                                                  y_mu=y_mu,
                                                  s=s)
            # y_mu updated
            mu = np.dot(h_matrix, self.output_weight[s])
            y_mu -= mu
        self.output_weight[np.isnan(self.output_weight)] = 0.0

    def fit_step(self, h_matrix, s: int,  y_mu=None):
        """
        Each  step of the fit process.
        :param h_matrix:
        :param int s: element of the ensemble.
        :param y_mu:
        :return:
        """
        izq = np.eye(h_matrix.shape[1]) * self.reg + np.dot(h_matrix.T, h_matrix)
        der = np.dot(h_matrix.T, y_mu)
        output_weight_s = solver(a=izq, b=der)
        return output_weight_s

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.
        :param numpy.array test_data: array like.
        :return: f(X) vector.
        """
        h_matrix = self.get_h_matrix(data=test_data)
        indicator = np.sum([self.alpha[s] * np.dot(h_matrix,
                                                   self.output_weight[s])
                            for s in range(self.size)], axis=0)
        return indicator
