from ..generic.predictor import Predictor
from ..util import solver
import numpy as np


class OLS(Predictor):
    """
    Ordinary Least Squares.
    """
    __name__ = 'Ordinary Least Squares'

    def instance_param_(self, train_data, train_target, parameter=None):
        """
        Instance parameters from dict.

        :param numpy.matrix train_data:
        :param numpy.array train_target:
        :param dict parameter: left for compatibility.
        :return:
        """
        self.train_target = train_target
        self.train_data = train_data
        self.n = train_data.shape[0]  # Number of instances
        self.dim = train_data.shape[1]  # Original dimension
        self.target_manager(train_target)

    def fit(self, train_data, train_target):
        """
        Use some train (data and target) and parameter to
        fit the classifier and construct the rules.

        :param numpy.array train_data: data with features.
        :param numpy.array train_target: targets in j codification.
        """
        self.instance_param_(train_data=train_data,
                             train_target=train_target)

        left = np.dot(train_data.T, train_data)
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
