from .adaboost import AdaBoostELM, EPS
import numpy as np


class AdaBoostNCELM(AdaBoostELM):
    """
    AdaBoost Negative Correlation meta-algorithm applied to ELM.
    """
    __name__ = 'AdaBoost Negative Correlation ELM'
    lambda_ = 0.0  # lambda hyperparameter

    def fit_step(self, h_matrix, s):
        """
        Each  step of the fit process.

        :param numpy.matrix h_matrix:
        :param int s:
        :return :
        """
        # Beta is calculated
        weight_matrix = np.diag(self.weight)
        izq = np.eye(h_matrix.shape[1]) + \
              self.reg * np.dot(h_matrix.T,
                              np.dot(weight_matrix,
                                     h_matrix))
        y_weighted = np.dot(weight_matrix, self.Y)
        der = np.dot(h_matrix.T, y_weighted)
        output_weight_s = np.linalg.solve(a=izq, b=der)

        y_pred = self.label_decoder_(np.dot(h_matrix,
                                            output_weight_s))

        # Ambiguity term
        if s == 0:
            amb = 0.0
        else:
            y_ensemble = self.predict(test_data=self.train_data)
            amb = 0.5 / s * np.invert(y_ensemble == y_pred).astype(np.float)
        pen = 1 - amb
        pen_lambda = np.power(pen, self.lambda_)

        # Error vector
        error_vector = np.invert(y_pred == self.train_target)
        e_s = np.sum(pen_lambda * self.weight * error_vector) \
              / np.sum(pen_lambda * self.weight) + EPS

        # Weight updated
        alpha_s = np.log((1 - e_s) / e_s) + np.log(self.labels + 1)
        self.alpha[s] = alpha_s
        self.weight = pen_lambda * self.weight * np.exp(alpha_s * error_vector)
        self.weight = self.weight / np.sum(self.weight)

        return output_weight_s
