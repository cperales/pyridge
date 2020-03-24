from ..generic.predictor import Predictor
import numpy as np
from ..util import solver


class RidgeRegressor(Predictor):
    """
    Ridge regressor.
    """
    __name__ = 'Ridge Regressor'

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

        left = np.eye(self.dim) * self.reg + \
               np.dot(train_data.T, train_data)
        right = np.dot(train_data.T, self.Y)

        self.output_weight = solver(a=left, b=right)

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: indicator.
        """
        indicator = np.dot(test_data, self.output_weight)
        return indicator
