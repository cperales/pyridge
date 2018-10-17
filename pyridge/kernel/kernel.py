import numpy as np
from pyridge.generic.kernel import KernelMethod, kernel_dict
from functools import partial
import logging

logger = logging.getLogger('pyridge')


class KernelRidge(KernelMethod):
    """
    Kernel Ridge classifier.
    """
    __name__ = 'Kernel Ridge'

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

        self.kernel_fun = partial(kernel_dict[self.kernel], self.gamma)
        self.train_data = train_data

        omega_train = self.kernel_fun(X=self.train_data)
        n = train_data.shape[0]
        izq = np.eye(n, dtype=np.float64) + self.reg * omega_train
        self.output_weight = np.linalg.solve(a=izq, b=self.Y)

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical
        rules.
        :param numpy.array test_data: matrix of data to predict.
        :return: matrix.
        """
        omega_test = self.kernel_fun(X=self.train_data, Y=test_data)
        indicator = np.dot(omega_test.transpose(), self.output_weight)
        return indicator