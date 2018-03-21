import logging

logger = logging.getLogger('PyRidge')


class Classifier(object):
    __name__ = 'Base classifier'

    def __init__(self, parameters=None):
        """
        :param dict parameters: dictionary with the parameters
            needed for training.
        """
        if parameters is not None:
            self.set_params(parameters)
        logger.debug('{} instanced'.format(self.__name__))

    def fit(self, train_data, train_target):
        """
        Use some train (data and target) and parameters to fit
        the classifier and construct the rules.

        :param numpy.array train_data: data with features.
        :param numpy.array train_target: targets in j codification.
        """
        pass

    def predict(self, test_data):
        """
        :param numpy.array test_data: array like.
        :return: predicted labels.
        """
        pass

    def set_params(self, parameters):
        """
        :param dict parameters: parameters in a dict.
        """
        pass
