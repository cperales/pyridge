from pyridge.generic.neural import NeuralMethod
import numpy as np


class NeuralRidge(NeuralMethod):
    """
    Neural Ridge classifier, also known as Extreme Learning Machine.
    It works as a single hidden layer neural network where
    neuron's weights are chosen randomly.
    """
    __name__ = 'Neural Ridge'

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

        self.get_weight_bias_()
        h_matrix = self.get_h_matrix_(data=train_data)

        izq = np.eye(h_matrix.shape[1]) + \
              self.reg * np.dot(h_matrix.T, h_matrix)
        der = np.dot(h_matrix.T, self.Y)

        self.output_weight = np.linalg.solve(a=izq, b=der)

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        h_matrix = self.get_h_matrix_(data=test_data)
        indicator = np.dot(h_matrix, self.output_weight)
        return indicator
