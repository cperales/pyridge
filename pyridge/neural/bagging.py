from .boosting import BoostingRidgeELM
import numpy as np


class BaggingELM(BoostingRidgeELM):
    """
    Bagging implemented to Neural Extreme Learning Machine.
    """
    __name__ = 'Bagging ELM'
    prop: float = 0.75  # Proportion of the dataset, 0.0 < prop < 1.0

    def fit_step(self, h_matrix, s):
        """
        Fit with part of the data from the whole set.
        This proportion can be given in the parameter dict;
        if not, proportion is 75%.

        :param h_matrix:
        :param int s:
        :return:
        """
        length = int(self.prop * h_matrix.shape[0])
        index = np.random.choice(h_matrix.shape[0],
                                 length)
        h = h_matrix[index]
        y = self.Y[index]
        izq = np.eye(h.shape[1]) + self.reg * np.dot(h.T, h)
        der = np.dot(h.T, y)
        output_weight_s = np.linalg.solve(a=izq, b=der)
        return output_weight_s
