from pyridge.generic.tensor import TensorELM, _TOL_
from pyridge.util import solver
import numpy as np
import logging

logger = logging.getLogger('pyridge')


class NegativeCorrelationELM(TensorELM):
    """
    Iterative Negative Correlation with Sherman-Morrison, updated.
    """
    __name__ = 'Negative Correlation ELM updated'
    # Negative Correlation
    lambda_: float
    max_iter_: int
    inv_left = None
    right = None
    I = None
    # For plotting
    list_norm = list

    def fit(self, train_data, train_target, parameter: dict):
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
        self.get_weight_bias_()
        # Simple ELMs
        self.H = np.array([self.get_h_matrix(data=train_data, s=s) for s in range(self.size)])
        self.I = np.eye(self.hidden_neurons)
        self.inv_left = np.array([solver(a=self.I / self.reg + np.dot(self.H[s].T, self.H[s]),
                                         b=self.I)
                                  for s in range(self.size)])
        self.right = np.array([np.dot(self.H[s].T, self.Y) for s in range(self.size)])
        self.output_weight = np.array([np.dot(self.inv_left[s], self.right[s]) for s in range(self.size)])
        beta_prev = np.copy(self.output_weight)

        # Multiple ELMs
        self.list_norm = list()
        norm = self.t * self.hidden_neurons * self.n
        # self.list_norm.append(norm)
        iter_: int = 0
        while norm > _TOL_ and iter_ < self.max_iter_:
            F = self.get_f()
            self.output_weight = np.array([np.array([np.dot(self.get_inv_left(f_j=F[:, j], s=s),
                                                            self.right[s, :, j])
                                                     for j in range(self.t)]).T
                                           for s in range(self.size)])
            norm = np.abs(beta_prev - self.output_weight).sum()
            self.list_norm.append(norm)
            beta_prev = np.copy(self.output_weight)
            iter_ += 1

    def get_inv_left(self, f_j, s):
        """

        :param f_j:
        :param int s:
        :return:
        """
        A_inv = self.inv_left[s]
        v = np.dot(self.H[s].T, f_j)

        num = np.dot(A_inv, np.dot(np.dot(v, v.T), A_inv))
        dem = self.reg / self.lambda_ + np.dot(v.T, np.dot(A_inv, v))

        return A_inv - num / dem
