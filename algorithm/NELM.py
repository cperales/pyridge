from generic import NeuralMethod


class NELM(NeuralMethod):
    """
    Neural Extreme Learning Machine
    """
    def fit(self, train, parameters):
        """

        :param train: input data has a dictionary structure, with keys 'data' and 'target'
        :param parameters: also dictionary structure
        :return:
        """
        self.t = train['target'].shape[1]
        self.hidden_neurons = parameters['hidden_neurons'] if parameters['hidden_neurons'] != 0 else self.t
