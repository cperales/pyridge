from pyridge.util.activation import activation_dict, nn_activation_dict
from pyridge.experiment.check import check_algorithm


def test_nn_activation():
    hyperparameter_nn = {'max_iter': [100],
                         'activation': ['sigmoid'],
                         'hidden_neurons': [5],
                         'solver': ['irprop'],
                         'batch_size': [150],
                         'learning_rate': [0.001]}
    for activation in nn_activation_dict.keys():
        hyperparameter_nn['activation'] = [activation]
        value_dict = check_algorithm(folder='data_regression',
                                     dataset='housing',
                                     algorithm='NeuralNetwork',
                                     hyperparameter=hyperparameter_nn,
                                     metric_list=['rmse'],
                                     classification=False)
        if value_dict['rmse'] > 0.4:
            raise ValueError('Activation function %s does not '
                             'provide good results', activation)


def test_activation():
    hyperparameter_elm = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10]}
    for activation in activation_dict.keys():
        hyperparameter_elm['activation'] = [activation]
        value_dict = check_algorithm(folder='data_regression',
                                     dataset='housing',
                                     algorithm='ELM',
                                     hyperparameter=hyperparameter_elm,
                                     metric_list=['rmse'],
                                     classification=False)
        if value_dict['rmse'] > 0.4:
            raise ValueError('Activation function %s does not '
                             'provide good results', activation)
