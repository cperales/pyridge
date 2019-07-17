from .elm import ELM
import numpy as np
from ..util import solver, accuracy, disagreement


class BaggingStepwiseELM(ELM):
    """
    Bagging Stepwise ELM Ensemble.
    """
    __name__ = "Bagging Stepwise ELM Ensemble"
    size: int
    m: int = 3
    prop: float = 0.75

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
        self.output_weight = np.zeros((self.size, self.hidden_neurons, self.Y.shape[1]))

        # Train the model
        h_matrix = self.get_h_matrix(data=train_data)
        self.output_weight[0] = self.fit_step(h=h_matrix, y=self.Y)
        acc = accuracy(clf=self,
                       pred_data=train_data,
                       real_targ=train_target)
        dis = 1.0
        expected_size = self.size

        removed = list()
        for s in range(1, self.size):
            # Random subset
            length = int(self.prop * self.n)
            index = np.random.choice(self.n, length)
            data_d = train_data[index]
            y_d = self.Y[index]
            h_matrix_d = self.get_h_matrix(data=data_d)
            # Train the model
            self.output_weight[s] = self.fit_step(h=h_matrix_d, y=y_d)
            new_acc = accuracy(clf=self,
                               pred_data=train_data,
                               real_targ=train_target)
            new_dis = disagreement(clf=self,
                                   pred_data=train_data,
                                   real_targ=train_target,
                                   S=s+1)
            if new_acc < acc or new_dis > dis:
                self.output_weight[s] = np.zeros(( self.hidden_neurons, self.Y.shape[1]))
                expected_size -= 1
                removed.append(s)
            else:
                dis = new_dis
                acc = new_acc
        self.output_weight = np.delete(self.output_weight, removed, axis=0)
        self.size = expected_size

    def fit_step(self, h, y):
        """
        Fit with part of the data from the whole set.

        :param h:
        :param y:
        :return:
        """
        left = np.eye(h.shape[1]) + self.reg * np.dot(h.T, h)
        right = np.dot(h.T, y)
        output_weight = solver(a=left, b=right)
        return output_weight

    def get_indicator(self, test_data, s=None):
        """
        Once instanced, classifier can predict test target
        from test data, using some mathematical rules.
        Valid for other ensembles.

        :param numpy.array test_data: array like.
        :param s:
        :return: predicted labels.
        """
        h_matrix = self.get_h_matrix(data=test_data)
        if s is None:
            indicator = np.mean([np.dot(h_matrix,
                                        self.output_weight[s])
                                 for s in range(len(self.output_weight))], axis=0)
        else:
            indicator = np.dot(h_matrix, self.output_weight[s])
        return indicator

    def predict_classifier(self, test_data, s=None):
        """
        Predict the label.

        :param numpy.array test_data: array like.
        :param s:
        :return: predicted labels.
        """
        indicator = self.get_indicator(test_data, s=s)
        predicted_labels = self.label_decoder(indicator)
        return predicted_labels

    def predict_regressor(self, test_data, s=None):
        """
        Predict the value (for linear).

        :param numpy.array test_data: array like.
        :param s:
        :return: predicted values.
        """
        return self.get_indicator(test_data, s=s)
