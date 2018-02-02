class Classifier(object):
    def __init__(self, parameters):
        """
        :param dict parameters: dictionary with the parameters needed for training.
        """
        pass

    def fit(self, train_data, train_target):
        """
        Use some train (data and target) and parameters to fit the classifier and construct the rules.

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
