class Classifier(object):

    method_param = {}

    def __init__(self):
        pass

    def fit(self, train, parameters):
        """
        Use some train (data and target) and parameters to fit the classifier and construct the rules.

        :param dict train: dictionary with two keys: 'data', with the features, and 'target' with an
            array of the labels.

        :param dict parameters: dictionary with the parameters needed for training.
        """
        pass

    def predict(self, test_data):
        """
        :param test_data: array like.
        :return:
        """
        pass
