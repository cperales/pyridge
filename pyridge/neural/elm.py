from ..generic.neural import NeuralMethod
import numpy as np
from ..util import solver


class ELM(NeuralMethod):
    """
    Neural Ridge classifier, also known as Extreme Learning Machine.
    It works as a single hidden layer neural network where
    neuron's weights are chosen randomly.
    """
    __name__ = 'Neural ELM'

    def instance_weight(self, train_data, train_target, parameter):
        """
        Instance parameters and get weight of neurons.

        :param train_data:
        :param train_target:
        :param parameter:
        :return:
        """
        self.instance_param_(train_data=train_data,
                             train_target=train_target,
                             parameter=parameter)
        self.get_weight_bias_()
        h_matrix = self.get_h_matrix(data=train_data)
        return h_matrix

    def fit(self, train_data, train_target, parameter):
        """
        Use some train (data and target) and parameter to
        fit the classifier and construct the rules.

        :param numpy.array train_data: data with features.
        :param numpy.array train_target: targets in j codification.
        :param dict parameter:
        """
        h_matrix = self.instance_weight(train_data=train_data,
                                        train_target=train_target,
                                        parameter=parameter)

        left = np.eye(self.hidden_neurons) / self.reg + \
               np.dot(h_matrix.T, h_matrix)
        right = np.dot(h_matrix.T, self.Y)

        self.output_weight = solver(a=left, b=right)

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: indicator.
        """
        indicator = np.dot(self.get_h_matrix(data=test_data),
                           self.output_weight)
        return indicator
