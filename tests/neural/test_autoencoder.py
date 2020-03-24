from pyridge.experiment.check import check_algorithm, create_logger


def test_nn_autoencoder():
    hyperparameter_nn = {'max_iter': [200],
                         'activation': ['sigmoid'],
                         'hidden_neurons': [2],
                         'solver': ['irprop'],
                         'batch_size': [50],
                         'learning_rate': [0.01]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='NeuralNetwork',
                                 hyperparameter=hyperparameter_nn,
                                 metric_list=['rmse'],
                                 autoencoder=True)
