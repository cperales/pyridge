import numpy as np
from ..neural.rnn import RandomNeuralNetwork
from .nc_nn import NegativeCorrelationNN
import logging

logger = logging.getLogger('pyridge')


class NegativeCorrelationRNN(NegativeCorrelationNN):
    """
    Negative Correlation for Random
    Neural Networks.
    """
    __name__: str = 'Negative Correlation ensemble ' \
                    'for Random Neural Networks'

    def fit(self, train_data, train_target, parameter):
        """
        Train several neural networks and update them with a negative
        correlation penalty.

        :param train_data: numpy.array with data (instances and features).
        :param train_target: numpy.array with the target to predict.
        :param dict parameter: keys
            - max_iter: number of iterations for training.
            - hidden_neurons: number of neurons in the hidden layer.
            - learning_rate: step to add in each iteration.
            - lambda_: coefficient for negative correlation penalty.
        """
        self.instance_param_(train_data=train_data,
                             train_target=train_target,
                             parameter=parameter)
        self.base_learner = np.empty(self.size, dtype=RandomNeuralNetwork)
        for s in range(self.size):
            self.base_learner[s] = RandomNeuralNetwork(classification=self.__classification__,
                                                       logging=False)
            self.base_learner[s].initial(train_data=train_data,
                                         train_target=train_target,
                                         parameter=parameter)

        for iteration in range(self.max_iter):
            # logger.debug('Iteration %i', iteration)
            f_bar = self.get_indicator(self.train_data)
            for s in range(self.size):
                # logger.debug('Neural Network %i', s)
                f_s = self.base_learner[s].get_indicator(self.train_data)
                penalty = - self.lambda_ * (f_s - f_bar)
                self.base_learner[s].solver(penalty=penalty)
