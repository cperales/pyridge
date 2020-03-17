from ..util.target_encode import id_encoder, id_decoder
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import logging

logger = logging.getLogger('pyridge')


class Predictor(object):
    __name__ = 'Base predictor'
    __base_type__: str
    labels: int
    t: int
    dim: int
    n: int
    reg: float
    train_target = None
    train_data = None
    label_encoder = None
    label_decoder = None
    Y = None
    output_weight = None

    def __init__(self, classification: bool = True, logging: bool = True):
        self.__classification__ = classification  # Default it works as a classifier
        if self.__classification__ is False:  # It works as a regressor
            self.label_encoder = id_encoder
            self.label_decoder = id_decoder
            self.target_manager = self.target_regression
            self.predict = self.predict_regressor
        else:
            self.label_binarizer_ = LabelBinarizer(neg_label=0, pos_label=1)
            self.target_manager = self.target_classification
            self.predict = self.predict_classifier
        # if logging is True:
        #     logger.debug('{} instanced'.format(self.__name__))

    def instance_param_(self, train_data, train_target, parameter):
        """
        Instance parameters from dict.

        :param numpy.matrix train_data:
        :param numpy.array train_target:
        :param dict parameter:
        :return:
        """
        self.train_target = train_target
        self.train_data = train_data
        self.n = train_data.shape[0]  # Number of instances
        self.dim = train_data.shape[1]  # Original dimension
        self.target_manager(train_target)

        # Instance the parameter dictionary
        self.__dict__.update(parameter)

    def target_regression(self, train_target):
        """
        :param train_target:
        :return:
        """
        self.Y = self.label_encoder(train_target)
        self.t = self.Y.shape[1]

    def target_classification(self, train_target):
        """
        :param train_target:
        :return:
        """
        self.label_binarizer_.fit(train_target)
        self.label_encoder = self.label_binarizer_.transform
        self.label_decoder = self.label_binarizer_.inverse_transform

        self.labels = np.unique(self.train_target).shape[0]
        self.Y = self.label_encoder(train_target).astype(np.float)
        self.t = self.Y.shape[1]

    def get_indicator(self, test_data):
        """
        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = None
        return indicator

    def predict_classifier(self, test_data):
        """
        Predict the label.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = self.get_indicator(test_data)
        predicted_labels = self.label_decoder(indicator)
        return predicted_labels

    def predict_proba(self, test_data):
        """
        Predict the probability.of class.

        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = self.get_indicator(test_data)
        predicted_prob = indicator / indicator.sum()
        return predicted_prob

    def predict_regressor(self, test_data):
        """
        Predict the value (for linear).

        :param numpy.array test_data: array like.
        :return: predicted values.
        """
        return self.get_indicator(test_data)
