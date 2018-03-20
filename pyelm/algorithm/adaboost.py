import numpy as np
from pyelm.algorithm.nelm import NELM
from pyelm.utils.target_encode import j_renorm


class AdaBoostNELM(NELM):
    """
    AdaBoost meta-algorithm applied to Neural Extreme Learning
    Machine.
    """
    __name__ = 'AdaBoost neural network'

    weight = None
    alpha = None
    beta_ensemble = None

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
        self.weight = np.ones(n) / n
        bias_matrix = np.resize(self.bias_vector.transpose(),
                                (n, h)).transpose()
        # h x n
        temp_H = np.dot(self.input_weight,
                        train_data.transpose()) + bias_matrix
        # n x h
        H = self.neuron_fun(temp_H.transpose())
        self.beta_ensemble = []

        for s in range(self.ensemble_size):
            beta_s = self.fit_step(H=H, train_target=train_target)
            self.beta_ensemble.append(beta_s)

    def fit_step(self, H, train_target):
        """
        :param numpy.array H: matrix that symbolises connection
            from input neurons to the ones from the hidden layer.
        :param numpy.array train_target: target from the training data.
        :return: beta matrix for each iteration of the s ensemble.
        """
        weight_matrix = np.diag(self.weight)
        H_reg = np.eye(H.shape[1]) / self.C + np.dot(np.dot(H.transpose(),
                                                            weight_matrix), H)
        beta_s = np.linalg.solve(H_reg,
                                 np.dot(np.dot(H.transpose(), weight_matrix),
                                        train_target))

        # Calculate errors
        y_hat = np.dot(H, beta_s)

        error_vector = (j_renorm(y_hat) == train_target).min(axis=1)
        e_s = (self.weight * error_vector).sum() / self.weight.sum()
        alpha_s = np.log((1 - e_s) / e_s) + np.log(beta_s.shape[1] - 1)
        self.weight = self.weight * np.exp(alpha_s * error_vector)
        self.weight = self.weight / self.weight.sum()  # Normalize
        return beta_s

    def predict(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.

        :param numpy.array test_data: matrix of data to predict.
        :return: matrix of the predicted targets.
        """
        n = test_data.shape[0]  # Number of instances to classify
        # h x n
        bias_matrix = np.resize(self.bias_vector.transpose(),
                                (n, self.hidden_neurons)).transpose()
        temp_H = np.dot(self.input_weight,
                        test_data.transpose()) + bias_matrix  # h x n
        H = self.neuron_fun(temp_H.transpose())

        test_target = np.empty((n, self.t))
        for s in range(self.ensemble_size):
            beta_s = self.beta_ensemble[s]
            indicator = np.dot(H, beta_s)
            y_hat = j_renorm(indicator)
            test_target += y_hat

        test_target = j_renorm(test_target)
        return test_target
