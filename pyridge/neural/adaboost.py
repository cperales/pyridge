from ..boosting import BoostingRidgeELM
from ..util import solver
import numpy as np

_EPS_ = 10**-12  # Avoid INF problems

classification_adaboost = ['AdaBoost Neural ELM',
                           'AdaBoost Negative Correlation Neural ELM']


class AdaBoostELM(BoostingRidgeELM):
    """
    AdaBoost meta-algorithm applied to Neural ELM.
    """
    __name__ = 'AdaBoost Neural ELM'
    weight = None
    positiveness = float

    def __init__(self, classification: bool = True):
        if classification is False:
            raise ValueError('This algorithm cannot be set '
                             'for regression problems')
        self.positiveness = 0.0
        super().__init__(classification=classification)

    def adaboost_weight(self, h_matrix, f_pred, s):
        """

        :param h_matrix:
        :param f_pred:
        :param s:
        :return: error vector
        """
        # Error vector
        error_vector = self.error_function(f_pred=f_pred, y=self.train_target)
        e_s = (self.weight * error_vector).sum()  # / self.weight.sum()  # Already normalized
        alpha_s = np.log((1.0 - e_s) / (e_s + _EPS_) + _EPS_) + self.positiveness

        # Weight
        weight = self.weight * np.exp(alpha_s * error_vector)
        weight = weight / weight.sum()

        return alpha_s, weight

    def fit(self, train_data, train_target, parameter):
        """
        Add array of weights and call super.

        :param train_data:
        :param train_target:
        :param parameter:
        :return:
        """
        h_matrix = self.instance_weight(train_data=train_data,
                                        train_target=train_target,
                                        parameter=parameter)
        self.weight = np.ones(self.n) / self.n
        self.alpha = np.empty(self.size)

        if self.__classification__ is True:
            self.positiveness = np.log(self.labels - 1.0)

        self.output_weight = np.zeros((self.size, self.hidden_neurons, self.t))
        for s in range(self.size):
            self.output_weight[s] = self.fit_step(h_matrix=h_matrix, s=s)
        self.output_weight[np.isnan(self.output_weight)] = 0.0

        # # Correction for RMSE in classification
        # self.alpha = self.alpha / self.alpha.sum()

    def fit_step(self, h_matrix, s: int,  y_mu=None):
        """
        Each  step of the fit process.

        :param h_matrix:
        :param int s: element of the ensemble.
        :param y_mu:
        :return:
        """
        weight_matrix = np.diag(self.weight)
        left = np.eye(self.hidden_neurons) / self.reg + \
               np.dot(h_matrix.T,
                      np.dot(weight_matrix, h_matrix))
        y_weighted = np.dot(weight_matrix, self.Y)
        right = np.dot(h_matrix.T, y_weighted)
        output_weight_s = solver(a=left, b=right)
        f_pred = np.dot(h_matrix, output_weight_s)

        alpha_s, weight = self.adaboost_weight(h_matrix=h_matrix, f_pred=f_pred, s=s)

        # Update weight and alpha
        self.alpha[s] = alpha_s
        self.weight = weight

        return output_weight_s

    def error_function(self, f_pred, y):
        """

        :param f_pred:
        :param y:
        :return: error_vector
        """
        y_pred = self.label_decoder(f_pred)
        error_vector = np.array(np.invert(y_pred == y), dtype=float)
        return error_vector

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

    def predict_classifier(self, test_data):
        """
        Following SAMME algorithm.

        :param test_data:
        :return:
        """
        h_matrix = self.get_h_matrix(data=test_data)
        indicator_matrix = [self.label_encoder(self.label_decoder(np.dot(h_matrix, self.output_weight[s])))
                            for s in range(self.size)]
        indicator = np.sum([self.alpha[s] * indicator_matrix[s] for s in range(self.size)], axis=0)
        predicted_labels = self.label_decoder(indicator)
        return predicted_labels
