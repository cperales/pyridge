from pyridge.experiment.check import check_algorithm


def test_nn():
    hyperparameter_nn = {'max_iter': [200],
                         'activation': ['sigmoid'],
                         'hidden_neurons': [10*i for i in range(1, 3)],
                         'solver': ['irprop'],
                         'batch_size': [50],
                         'learning_rate': [0.01]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='NeuralNetwork',
                                 hyperparameter=hyperparameter_nn,
                                 metric_list=['accuracy', 'rmse'])


def test_nn_regression():
    hyperparameter_nn = {'max_iter': [200],
                         'activation': ['sigmoid'],
                         'hidden_neurons': [5, 10, 15],
                         'solver': ['irprop'],
                         'batch_size': [150],
                         'learning_rate': [0.001]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='NeuralNetwork',
                                 hyperparameter=hyperparameter_nn,
                                 metric_list=['rmse'],
                                 classification=False)
