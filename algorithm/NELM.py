from generic import NeuralMethod
from utility.target_encode import *
import numpy as np
import logging


class NELM(NeuralMethod):
    """
    Neural Extreme Learning Machine
    """
    def __init__(self):
        logging.debug('Neural Extreme Learning Machine instanced')

    def fit(self, train, parameters):
        self.t = train['target'].shape[1]
        self.hidden_neurons = parameters['hidden_neurons'] if parameters['hidden_neurons'] != 0 else self.t
        self.C = parameters['C']

        n = train['data'].shape[0]
        m = train['data'].shape[1]
        h = self.hidden_neurons

        self.input_weight = np.random.rand(h, m)  # h x m
        self.bias_vector = np.random.rand(h, 1)  # h x 1
        # bias_matrix = np.resize(self.bias_vector, (h, n)).transpose()  # h x n
        bias_matrix = np.resize(self.bias_vector.transpose(), (n, h)).transpose()
        temp_H = np.dot(self.input_weight, train['data'].transpose()) + bias_matrix  # h x n
        H = self.neuron_fun(temp_H.transpose())  # n x h

        if self.C == 0:  # No regularization
            # inv_H = np.linalg.pinv(H)
            inv_H = np.linalg.inv(H)
            self.output_weight = np.dot(inv_H, train['target'])
        else:
            alpha = np.eye(H.shape[0]) / self.C + np.dot(H, H.transpose())
            # inv_alpha = np.linalg.pinv(alpha)
            inv_alpha = np.linalg.inv(alpha)
            self.output_weight = np.dot(H.transpose(), np.dot(inv_alpha, train['target']))

    def predict(self, test_data):
        n = test_data.shape[0]  # Number of instances to classify
        # bias_matrix = np.resize(self.bias_vector, (self.hidden_neurons, n)).transpose()  # h x n
        bias_matrix = np.resize(self.bias_vector.transpose(), (n, self.hidden_neurons)).transpose()
        temp_H = np.dot(self.input_weight, test_data.transpose()) + bias_matrix  # h x n
        H = self.neuron_fun(temp_H.transpose())
        indicator = np.dot(H, self.output_weight)
        test_target = j_renorm(indicator)
        return test_target

    def save_clf_param(self):
        return self.__dict__

    def load_clf_param(self, clf_param):
        self.__dict__ = clf_param
