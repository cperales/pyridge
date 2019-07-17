from .adaboost import AdaBoostELM, _EPS_
import numpy as np


class AdaBoostNCELM(AdaBoostELM):
    """
    AdaBoost Negative Correlation meta-algorithm applied to Neural ELM.
    """
    __name__ = 'AdaBoost Negative Correlation Neural ELM'
    lambda_: float  # lambda hyperparameter

    def adaboost_weight(self, h_matrix, f_pred, s):
        """

        :param h_matrix:
        :param f_pred:
        :param s:
        :return:
        """
        error_vector = self.error_function(f_pred=f_pred, y=self.train_target)
        # Ambiguity term
        if s == 0:
            pen_lambda = 1.0
        else:
            f_ensemble = np.sum([self.alpha[t] * np.dot(h_matrix,
                                                        self.output_weight[t])
                                 for t in range(s)], axis=0)
            # y_ensemble = self.predict_s(test_data=self.train_data)
            error_ensemble = self.error_function(f_pred=f_ensemble, y=self.train_target)
            amb = 0.5 * np.mean(error_ensemble - error_vector)
            pen = 1.0 - np.abs(amb)
            pen_lambda = np.power(pen, self.lambda_)

        # Error vector
        e_s = np.sum(pen_lambda * self.weight *
                     error_vector) / np.sum(pen_lambda * self.weight)
        alpha_s = np.log((1.0 - e_s) / (e_s + _EPS_) + _EPS_) + self.positiveness

        # Weight
        weight = self.weight * np.exp(alpha_s * error_vector) * pen_lambda
        weight = weight / weight.sum()

        return alpha_s, weight
