from .boosting import BoostingRidgeELM
import numpy as np


class DiverseELM(BoostingRidgeELM):
    """
    Diverse Neural Extreme Learning Machine.
    """
    __name__ = 'Diverse ELM'
    M_k = None
    div: float = 1.0

    def fit_step(self, h_matrix, s):
        """
        Each  step of the fit process.

        :param h_matrix:
        :param int s:
        :return :
        """
        if s == 0:
            self.M_k = np.zeros((self.labels, self.hidden_neurons, self.hidden_neurons))
        else:
            beta = self.output_weight[s - 1]
            u = beta / np.linalg.norm(beta, ord=2, axis=0)
            for j in range(u.shape[1]):
                u_j = u[:, j].reshape((self.hidden_neurons, 1))
                self.M_k[j] += np.dot(u_j, u_j.T)

        output_weight_s = np.zeros((self.hidden_neurons, self.Y.shape[1]))
        izq = self.reg * np.eye(h_matrix.shape[1]) + \
              np.dot(h_matrix.T, h_matrix)
        der = np.dot(h_matrix.T, self.Y)
        if s != 0:
            for j in range(self.Y.shape[1]):
                izq_j = izq + (self.div + self.train_data.shape[0] / s) * self.M_k[j]
                der_j = der[:, j]
                output_weight_s[:, j] = np.linalg.solve(a=izq_j, b=der_j)
        else:
            output_weight_s = np.linalg.solve(a=izq, b=der)

        return output_weight_s
