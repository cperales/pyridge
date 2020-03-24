from .linear import RidgeRegressor
import numpy as np
from ..util import solver


class BoostingRidgeRegressor(RidgeRegressor):
    """
    Boosting Ridge ensemble applied to Linear Regressor.
    """
    __name__ = 'Boosting Ridge'
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
        self.instance_param_(train_data=train_data,
                             train_target=train_target,
                             parameter=parameter)
        self.alpha = np.ones(self.size)
        y_mu = self.Y.copy()
        self.output_weight = np.zeros((self.size, self.dim, self.Y.shape[1]))
        for s in range(self.size):
            self.output_weight[s] = self.fit_step(x=train_data,
                                                  y_mu=y_mu)
            # y_mu updated
            y_mu -= self.get_indicator(train_data)
        self.output_weight[np.isnan(self.output_weight)] = 0.0

    def fit_step(self, x, y_mu=None):
        """
        Each  step of the fit process.

        :param x:
        :param y_mu:
        :param int s: element of the ensemble, for inheritance.
        :return:
        """
        izq = np.eye(x.shape[1]) * self.reg + np.dot(x.T, x)
        der = np.dot(x.T, y_mu)
        output_weight_s = solver(a=izq, b=der)
        return output_weight_s

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = np.sum([self.alpha[s] * np.dot(test_data,
                                                   self.output_weight[s])
                            for s in range(self.size)], axis=0)
        return indicator
