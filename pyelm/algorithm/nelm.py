from pyelm import logger_pyelm
from pyelm.utils.target_encode import *
from pyelm.generic import NeuralMethod


class NELM(NeuralMethod):
    """
    Neural Extreme Learning Machine. Neural Network's version of the Extreme Learning Machine,
    in which "first layer" neuron's weights are chosen randomly.
    """
    def __init__(self, parameters=None):
        """
        :param dict parameters: dictionary with the parameters needed for training. It must contain:

                - hidden_neurons: the number of the neurons in the hidden layer.
                - C: regularization of H matrix.
        """
        if parameters is not None:
            self.__call__(parameters)
        logger_pyelm.debug('Neural Extreme Learning Machine instanced')

    def fit(self, train_data, train_target):
        """
        Use some train (data and target) and parameters to fit the classifier and construct the rules.

        :param numpy.array train_data: data with features.
        :param numpy.array train_target: targets in j codification.
        """
        self.t = train_target.shape[1]
        
        n = train_data.shape[0]
        m = train_data.shape[1]
        h = self.hidden_neurons

        self.input_weight = np.random.rand(h, m)  # h x m
        self.bias_vector = np.random.rand(h, 1)  # h x 1
        bias_matrix = np.resize(self.bias_vector.transpose(), (n, h)).transpose()
        temp_H = np.dot(self.input_weight, train_data.transpose()) + bias_matrix  # h x n
        H = self.neuron_fun(temp_H.transpose())  # n x h

        if self.C == 0:  # Means no regularization
            H_inv = np.linalg.pinv(H)  # Usually np.linalg.solve gives an error
            self.output_weight = np.dot(H_inv, train_target)
        else:
            alpha = np.eye(H.shape[0]) / self.C + np.dot(H, H.transpose())
            self.output_weight = np.dot(H.transpose(), np.linalg.solve(alpha, train_target))

    def predict(self, test_data):
        """
        Once instanced, classifier can predict test target from test data, using some mathematical
        rules.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        n = test_data.shape[0]  # Number of instances to classify
        bias_matrix = np.resize(self.bias_vector.transpose(), (n, self.hidden_neurons)).transpose()
        temp_H = np.dot(self.input_weight, test_data.transpose()) + bias_matrix  # h x n
        H = self.neuron_fun(temp_H.transpose())
        indicator = np.dot(H, self.output_weight)
        test_target = j_renorm(indicator)
        return test_target

    def save_clf_param(self):
        return {'C': self.C,
                'hidden_neurons': self.hidden_neurons}

    def __call__(self, parameters):
        """
        :param dict parameters: dictionary with the parameters needed for training. It must contain:

                - hidden_neurons: the number of the neurons in the hidden layer.
                - C: regularization of H matrix.
        """
        self.hidden_neurons = parameters['hidden_neurons'] if parameters['hidden_neurons'] != 0 else self.t
        self.C = parameters['C']
