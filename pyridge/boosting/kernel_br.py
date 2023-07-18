from ..kernel.kelm import KernelELM, kernel_dict
import numpy as np
from ..util import solver
from functools import partial


class KernelBoostingRidgeELM(KernelELM):
    """
    Boosting Ridge ensemble applied to Kernel ELM.
    """
    __name__ = 'Boosting Ridge Kernel ELM'
    size: int
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
        Y_mu = self.Y.astype(np.float64)
        # Kernel
        self.kernel_fun = partial(kernel_dict[self.kernel], self.gamma)
        self.train_data = train_data

        omega_train = self.kernel_fun(X=self.train_data)
        n = train_data.shape[0]
        izq = np.eye(n, dtype=np.float64) * self.reg + omega_train
        self.output_weight = np.zeros((self.size, n, self.t))
        for s in range(self.size):
            self.output_weight[s] = solver(a=izq, b=Y_mu)
            # Y_mu update
            Y_mu -= np.dot(omega_train.T, self.output_weight[s])
        self.output_weight[np.isnan(self.output_weight)] = 0.0  # Avoid NaNs

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.
        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        omega_test = self.kernel_fun(X=self.train_data, Y=test_data)
        indicator = np.sum([self.alpha[s] * np.dot(omega_test.T,
                                                   self.output_weight[s])
                            for s in range(len(self.output_weight))], axis=0)
        return indicator
