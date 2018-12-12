import logging
from sklearn.preprocessing import LabelBinarizer
import numpy as np

logger = logging.getLogger('pyridge')


class Classifier(object):
    __name__ = 'Base classifier'
    labels: int
    dim: int
    n: int
    reg: float = 1.0
    train_target = None
    train_data = None
    label_encoder_ = None
    label_decoder_ = None
    Y = None
    output_weight = None

    def __init__(self):
        logger.debug('{} instanced'.format(self.__name__))
        self.label_encoder_ = LabelBinarizer(neg_label=0, pos_label=1)

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
        self.labels = len(np.unique(train_target))
        self.dim = train_data.shape[1]  # Original dimension
        self.n = train_data.shape[0]  # Number of instances

        # TODO: for probabilities in binary problems, other
        # transformation should be taking into account
        label_encoder_ = LabelBinarizer(neg_label=0,
                                        pos_label=1).fit(train_target)
        self.label_encoder_ = label_encoder_.transform
        self.label_decoder_ = label_encoder_.inverse_transform

        self.Y = self.label_encoder_(train_target).astype(np.float64)

        # Instance the parameter dictionary
        if 'ensemble_size' in parameter.keys():
            parameter.update({'size': parameter['ensemble_size']})
            parameter.pop('ensemble_size')
        self.__dict__.update(parameter)

    def get_indicator(self, test_data):
        """
        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = None
        return indicator

    def predict(self, test_data):
        """
        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        indicator = self.get_indicator(test_data)
        predicted_labels = self.label_decoder_(indicator)
        return predicted_labels
