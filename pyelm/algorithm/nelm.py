import numpy as np
from pyelm.utils.target_encode import j_renorm
from pyelm.generic import NeuralMethod


class NELM(NeuralMethod):
    """
    Neural Extreme Learning Machine. Neural Network's version
    of the Extreme Learning Machine, in which "first layer"
    neuron's weights are chosen randomly.
    """
    __name__ = 'Neural Extreme Learning Machine'

    def fit(self, train_data, train_target):
        """
        Use some train (data and target) and parameters to
        fit the classifier and construct the rules.

        :param numpy.array train_data: data with features.
        :param numpy.array train_target: targets in j codification.
        """
        self.t = train_target.shape[1]

        n = train_data.shape[0]
        m = train_data.shape[1]
        h = self.hidden_neurons

        # h x m
        self.input_weight = np.random.rand(h, m)
        # h x 1
        self.bias_vector = np.random.rand(h, 1)
        bias_matrix = np.resize(self.bias_vector.transpose(),
                                (n, h)).transpose()
        # h x n
        temp_H = np.dot(self.input_weight,
                        train_data.transpose()) + bias_matrix
        # n x h
        H = self.neuron_fun(temp_H.transpose())

        if self.C == 0:  # Means no regularization
            # Usually np.linalg.solve gives an error
            H_inv = np.linalg.pinv(H)
            self.output_weight = np.dot(H_inv, train_target)
        else:
            alpha = np.eye(H.shape[0]) / self.C + np.dot(H,
                                                         H.transpose())
            self.output_weight = np.dot(H.transpose(),
                                        np.linalg.solve(alpha,
                                                        train_target))

    def predict(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        # Number of instances to classify
        n = test_data.shape[0]
        bias_matrix = np.resize(self.bias_vector.transpose(),
                                (n, self.hidden_neurons)).transpose()
        # h x n
        temp_H = np.dot(self.input_weight,
                        test_data.transpose()) + bias_matrix
        H = self.neuron_fun(temp_H.transpose())
        indicator = np.dot(H, self.output_weight)
        test_target = j_renorm(indicator)
        return test_target

    def get_params(self, deep=False):
        """

        :param bool deep: If just wants the hyperparameters, `deep  = False`.
            For getting subobjects and methods, `deep = True`.
        :return: Parameters as a dictionary.
        """
        to_return = None
        to_return = {'C': self.C,
                     'hidden_neurons': self.hidden_neurons,
                     'ensemble_size': self.ensemble_size}
        if deep is True:
            to_return.update(self.__dict__)
        return to_return

    def set_params(self, parameters):
        """
        :param dict parameters: dictionary with the parameters
            needed for training. It must contain:

                - hidden_neurons: the number of the
                    neurons in the hidden layer.
                - C: regularization of H matrix.
                - ensemble_size: (optional) used for
                    meta algorithms as AdaBoost.
        """
        self.hidden_neurons = parameters['hidden_neurons'] if \
            parameters['hidden_neurons'] == 0 else self.t
        self.C = parameters['C']
        self.ensemble_size = parameters['ensemble_size'] if \
            'ensemble_size' in parameters else 1

    def __call__(self, parameters):
        """
        :param dict parameters: dictionary with the parameters
            needed for training. It must contain:

                - hidden_neurons: the number of the
                    neurons in the hidden layer.
                - C: regularization of H matrix.
                - ensemble_size: (optional) used for
                    meta algorithms as AdaBoost.
        """
        self.set_params(parameters)
