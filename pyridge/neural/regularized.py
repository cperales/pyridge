from .diverse import DiverseELM
import numpy as np
from ..util import solver


class RegularizedEnsembleELM(DiverseELM):
    """
    Regularized Extreme Learning Machine.
    """
    __name__ = 'Regularized Ensemble ELM'
    r: float  # Composition of alphas, 0 < r < 1
    b = None
    der = None

    def fit_step(self, h_matrix, y_mu=None, s:int=1):
        """
        Each step of the fit process.

        :param h_matrix:
        :param y_mu:
        :param int s:
        :return:
        """
        self.alpha[s] = (1 - self.r) * np.power(self.r, s) / \
                        (1 - np.power(self.r, self.size))
        if s == 0:
            self.der = np.dot(h_matrix.T, self.Y)
            izq = self.reg * np.eye(h_matrix.shape[1]) + \
                  np.dot(h_matrix.T, h_matrix)
            b_s = solver(a=izq, b=np.eye(izq.shape[0]))
            output_weight_s = np.dot(b_s, self.der)
            self.b = np.array([b_s] * self.Y.shape[1])
        else:
            output_weight_s = np.zeros((self.hidden_neurons, self.t))
            coef = self.reg / (self.n * self.alpha[s - 1])
            for j in range(self.Y.shape[1]):
                beta_s_j = self.output_weight[s - 1][:, 0]
                b_j = self.b[j]
                num = b_j * (np.dot(beta_s_j, beta_s_j.T))
                dem = coef * np.dot(beta_s_j, beta_s_j.T) + \
                      np.dot(beta_s_j.T,
                             np.dot(b_j, beta_s_j))
                self.b[j] = np.dot(np.eye(num.shape[0]) - num / dem, b_j)
                output_weight_s[:, j] = np.dot(self.b[j], self.der[:, j])

        return output_weight_s
