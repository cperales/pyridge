from .boosting import BoostingRidgeELM
import numpy as np

EPS = 10**-10


class AdaBoostELM(BoostingRidgeELM):
    """
    AdaBoost meta-algorithm applied to ELM.
    """
    __name__ = 'AdaBoost Neural ELM'
    weight = None

    def fit(self, train_data, train_target, parameter):
        """
        Add array of weights and call super.

        :param train_data:
        :param train_target:
        :param parameter:
        :return:
        """
        self.weight = np.ones(train_data.shape[0])
        super().fit(train_data=train_data,
                    train_target=train_target,
                    parameter=parameter)

    def fit_step(self, h_matrix, y_mu=None, s:int=1):
        """
        Each step of the fit process.

        :param h_matrix:
        :param y_mu:
        :param int s:
        :return:
        """
        weight_matrix = np.diag(self.weight)
        izq = np.eye(h_matrix.shape[1]) + \
              self.reg * np.dot(h_matrix.T,
                                np.dot(weight_matrix,
                                       h_matrix))
        y_weighted = np.dot(weight_matrix, self.Y)
        der = np.dot(h_matrix.T, y_weighted)
        output_weight_s = np.linalg.solve(a=izq, b=der)

        # Weight and alpha updated
        y_pred = self.label_decoder_(np.dot(h_matrix,
                                            output_weight_s))
        error_vector = np.invert(y_pred == self.train_target)
        e_s = np.sum(self.weight * error_vector) / np.sum(self.weight) + EPS
        alpha_s = np.log((1 - e_s) / e_s) + \
                  np.log(self.labels + 1)
        self.weight = self.weight * np.exp(alpha_s * error_vector)
        self.weight = self.weight / np.sum(self.weight)
        self.alpha[s] = alpha_s

        return output_weight_s
