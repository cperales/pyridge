from .boosting import BoostingRidgeELM
import numpy as np
from ..util import solver


class DiverseELM(BoostingRidgeELM):
    """
    Diverse Neural Extreme Learning Machine.
    """
    __name__ = 'Diverse Neural ELM'
    M_k = None
    div: float

    def fit_step(self, h_matrix, y_mu=None, s:int=1):
        """
        Each step of the fit process.

        :param h_matrix:
        :param y_mu:
        :param int s:
        :return:
        """
        if s == 0:
            self.M_k = np.zeros((self.t, self.hidden_neurons, self.hidden_neurons))
        else:
            beta = self.output_weight[s - 1]
            u = beta / np.linalg.norm(beta, ord=2, axis=0)
            for j in range(u.shape[1]):
                u_j = u[:, j].reshape((self.hidden_neurons, 1))
                self.M_k[j] += np.dot(u_j, u_j.T)

        output_weight_s = np.zeros((self.hidden_neurons, self.t))
        izq = self.reg * np.eye(self.hidden_neurons) + \
              np.dot(h_matrix.T, h_matrix)
        der = np.dot(h_matrix.T, self.Y)
        if s != 0:
            for j in range(self.Y.shape[1]):
                izq_j = izq + (self.div + self.n / s) * self.M_k[j]
                der_j = der[:, j]
                output_weight_s[:, j] = solver(a=izq_j, b=der_j)
        else:
            output_weight_s = solver(a=izq, b=der)

        return output_weight_s
