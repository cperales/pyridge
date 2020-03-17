from ..generic.neural import NeuralMethod
from ..util.activation import nn_activation_dict
import numpy as np
import logging

logger = logging.getLogger('pyridge')


class RandomNeuralNetwork(NeuralMethod):
    """
    Neural Network with one hidden layer
    and random projection.
    """
    __name__: str = 'Random Neural Network'
    max_iter: int  # Number of iterations
    learning_rate: float  # Step
    # Matrix
    input_weight: np.array
    output_weight: np.array
    bias_input_layer: np.array
    bias_output_layer: np.array
    temp_h: np.array  # Value of the hidden layer before applying activation.
    temp_o: np.array  # Value of the output layer before applying activation.
    # Neuronal functions
    activation = None
    activation_out = None
    activation_out_der = None
    # Solver
    solver = None
    # SGD
    batch_size: int
    # IRPROP+
    initial_step: float = 0.0125
    min_delta: float = 0.0
    max_delta: float = 5.0
    pos_eta: float = 1.2
    neg_eta: float = 0.2
    dE_dW_pre_output = None
    delta_output = None
    update_output = None
    error = None
    prev_error = None
    # Debug
    iteration: int = 0
    # Store information
    hidden_layer = None

    def get_layers(self):
        """
        Feed forward random assignation of the two layers.
        """
        self.get_input_layer()
        self.get_output_layer()

    def get_input_layer(self):
        """
        Weights and bias for input layer.
        """
        self.input_weight = np.random.random((self.dim,
                                              self.hidden_neurons)) * 2.0 - 1.0
        self.bias_input_layer = np.random.random((self.hidden_neurons, 1))

    def get_output_layer(self):
        self.output_weight = np.random.random((self.hidden_neurons,
                                               self.t)) * 2.0 - 1.0
        self.bias_output_layer = np.random.random((self.t, 1))

    def initial(self, train_data, train_target, parameter):
        """
        Instance parameters and initial layer.

        :param train_data:
        :param train_target:
        :param parameter:
        :return:
        """
        self.instance_param_(train_data=train_data,
                             train_target=train_target,
                             parameter=parameter)
        self.get_layers()
        self.solver = getattr(self, parameter['solver'])
        self.__name__ += ' with ' + parameter['solver']
        batch_size = parameter.get('batch_size', self.n)
        self.batch_size = batch_size if batch_size < self.n else self.n
        if parameter['solver'] == 'irprop':
            self.delta_output = np.ones((self.hidden_neurons, self.t)) * self.initial_step
            self.prev_error = np.inf
            self.dE_dW_pre_output = None

        act_dict = nn_activation_dict[parameter['activation']]
        self.activation = act_dict['activation']

        if self.__classification__ is True:
            self.activation_out = self.activation
            self.activation_out_der = act_dict['derivative']
        else:  # Linear activation in the output layer
            self.activation_out = nn_activation_dict['linear']['activation']
            self.activation_out_der = nn_activation_dict['linear']['derivative']

        # Instancing hidden layer with all the train data
        self.hidden_layer, output_layer = self.forward(self.train_data)

    def fit(self, train_data, train_target, parameter):
        """
        Train the neural network with gradient descent.

        :param train_data: numpy.array with data (instances and features).
        :param train_target: numpy.array with the target to predict.
        :param dict parameter: keys
            - max_iter: number of iterations for training.
            - neurons: number of neurons in the hidden layer.
            - learning_rate: step to add in each iteration.
        """
        self.initial(train_data=train_data,
                     train_target=train_target,
                     parameter=parameter)
        for self.iteration in range(self.max_iter):
            # logger.debug('Iteration = %i', self.iteration)
            self.solver(penalty=None)
        # logger.debug('')

    # The neural network get_indicators.
    def forward(self, test_data):
        hidden_layer = self.activation(np.dot(test_data, self.input_weight) +
                                       self.bias_input_layer.T)
        self.temp_o = np.dot(hidden_layer, self.output_weight) + self.bias_output_layer.T
        output_layer = self.activation_out(self.temp_o)
        return hidden_layer, output_layer

    def forward_train(self, hidden_layer):
        self.temp_o = np.dot(hidden_layer, self.output_weight) + self.bias_output_layer.T
        output_layer = self.activation_out(self.temp_o)
        return output_layer

    def get_indicator(self, test_data):
        """
        Predict value.

        :param test_data:
        :return:
        """
        _, output = self.forward(test_data)
        output[np.isnan(output)] = 0.0
        return output

    # Solvers
    def get_grads(self, error):
        """

        :param error
        :return:
        """
        # Gradient in output layer
        grad_output = error * self.activation_out_der(self.temp_o)
        return grad_output

    def get_dE_dW(self, grad_output, hidden_layer):
        """
        Compute de derivate of the Error respect to the weights.

        :param grad_output:
        :param hidden_layer:
        :return:
        """
        dE_dWeight_output = self.learning_rate * np.dot(hidden_layer.T, grad_output)
        return dE_dWeight_output

    def get_batches_random(self, penalty=None):
        """
        """
        if self.batch_size == self.n:
            index = np.ones(self.n, dtype=bool)
        else:
            index = np.random.choice(self.n, self.batch_size)
        hidden_layer = self.hidden_layer[index]
        target = self.Y[index]
        penalty_subset = penalty[index] if penalty is not None else None
        return hidden_layer, target, penalty_subset

    def backpropagation(self, penalty=None):
        """
        Adjust the weights after the prediction using a few instances
        chosen stochastically.

        :param penalty: for NC.
        """
        # Choosing stochastically a subset
        hidden_layer, Y_subset, penalty_subset = self.get_batches_random(penalty)
        # Error
        output_layer = self.forward_train(hidden_layer=hidden_layer)

        error = output_layer - Y_subset if penalty_subset is None \
            else output_layer - Y_subset + penalty_subset
        logger.debug('Iteration %i, error = %f', self.iteration, np.linalg.norm(error))
        # Gradients and dE / dW
        grad_output = self.get_grads(error=error)
        dE_dWeight_output = self.get_dE_dW(grad_output=grad_output,
                                           hidden_layer=hidden_layer)
        # Updates
        self.update(delta_weight_output=dE_dWeight_output)

    def sgd(self, penalty=None):
        self.backpropagation(penalty=penalty)

    def update(self, delta_weight_output):
        """
        Update for back propagation.

        :param delta_weight_output:
        :return:
        """
        # Update output layer
        self.bias_output_layer -= np.mean(delta_weight_output)
        self.output_weight -= delta_weight_output

    @staticmethod
    def get_sign(dEdW, prev_dEdW=None):
        sign = np.zeros_like(dEdW) if prev_dEdW is None \
            else np.sign([i * j for i, j in zip(dEdW, prev_dEdW)])
        return sign

    def irprop_pos_condition(self, delta, dE_dW):
        new_delta = delta * self.pos_eta
        new_delta[new_delta > self.max_delta] = self.max_delta
        update_weight = np.sign(dE_dW) * new_delta  # Update is attached with a minus
        return new_delta, update_weight

    def irprop_zero_condition(self, delta, dE_dW):
        update_weight = np.sign(dE_dW) * delta
        return delta, update_weight

    def irprop_neg_condition(self, delta, update_weight, diff_error):
        # Delta
        new_delta = delta * self.neg_eta
        new_delta[new_delta < self.min_delta] = self.min_delta
        # Update Weight
        new_update_weight = update_weight if diff_error > 0.0 \
            else np.zeros_like(update_weight)
        return new_delta, new_update_weight

    def irprop(self, penalty=None):
        """
        Algorithm iRPROP+ implementation.

        :param penalty:
        """
        # Choosing stochastically a subset
        hidden_layer, Y_subset, penalty_subset = self.get_batches_random(penalty)
        output_layer = self.forward_train(hidden_layer=hidden_layer)
        error = output_layer - Y_subset if penalty_subset is None \
            else output_layer - Y_subset + penalty_subset
        self.error = np.linalg.norm(error)
        # logger.debug('Error = %f', self.error)
        diff_error = self.error - self.prev_error
        # Gradients
        grad_output = self.get_grads(error=error)
        dE_dWeight_output = self.get_dE_dW(grad_output=grad_output,
                                           hidden_layer=hidden_layer)

        # # Output layer
        sign_out = self.get_sign(dEdW=dE_dWeight_output,
                                 prev_dEdW=self.dE_dW_pre_output)
        self.update_output = np.empty_like(self.output_weight)

        # Positive
        mask_out_pos = sign_out == 1
        delta_output_pos, update_output_pos = \
            self.irprop_pos_condition(dE_dW=dE_dWeight_output[mask_out_pos],
                                      delta=self.delta_output[mask_out_pos])
        self.delta_output[mask_out_pos] = delta_output_pos
        self.update_output[mask_out_pos] = update_output_pos

        # Zero
        mask_out_zero = sign_out == 0
        delta_output_zero, update_output_zero = \
            self.irprop_zero_condition(dE_dW=dE_dWeight_output[mask_out_zero],
                                       delta=self.delta_output[mask_out_zero])
        self.delta_output[mask_out_zero] = delta_output_zero
        self.update_output[mask_out_zero] = update_output_zero

        # Negative
        mask_out_neg = sign_out == -1
        delta_output_neg, update_output_neg = \
            self.irprop_neg_condition(delta=self.delta_output[mask_out_neg],
                                      update_weight=self.update_output[mask_out_neg],
                                      diff_error=diff_error)
        self.delta_output[mask_out_neg] = delta_output_neg
        self.update_output[mask_out_neg] = update_output_neg

        # self.output_weight += self.update_output
        self.dE_dW_pre_output = dE_dWeight_output

        # # Update
        self.update(delta_weight_output=self.update_output)
        self.prev_error = self.error
