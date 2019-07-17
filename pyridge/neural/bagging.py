from .elm import ELM
import numpy as np
from ..util import solver


class BaggingELM(ELM):
    """
    Bagging implemented to Neural Extreme Learning Machine.
    """
    __name__ = 'Bagging Neural ELM'
    size: int
    Y_mu = None
    I = None
    alpha = None
    prop: float = 0.75  # Proportion of the dataset, 0.0 < prop < 1.0

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
        self.alpha = np.ones(self.size) / self.size
        self.get_weight_bias_()
        h_matrix = self.get_h_matrix(data=train_data)

        self.output_weight = np.zeros((self.size, self.hidden_neurons, self.Y.shape[1]))
        self.I = np.eye(self.hidden_neurons)
        for s in range(self.size):
            self.output_weight[s] = self.fit_step(train_data=train_data,
                                                  train_target=train_target)
        self.output_weight[np.isnan(self.output_weight)] = 0.0

    def fit_step(self, train_data, train_target):
        """
        Fit with part of the data from the whole set.
        This proportion can be given in the parameter dict;
        if not, proportion is 75%.

        :param train_data:
        :param train_target:
        :return:
        """
        length = int(self.prop * self.n)
        index = np.random.choice(self.n, length)
        subset_X = train_data[index]

        H = self.get_h_matrix(data=subset_X)
        y = self.Y[index]
        izq = self.I / self.reg + np.dot(H.T, H)
        der = np.dot(H.T, y)
        output_weight_s = solver(a=izq, b=der)

        return output_weight_s

    def get_indicator(self, test_data):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        h_matrix = self.get_h_matrix(data=test_data)
        indicator = np.sum([self.alpha[s] * np.dot(h_matrix,
                                                   self.output_weight[s])
                            for s in range(len(self.output_weight))], axis=0)
        return indicator
