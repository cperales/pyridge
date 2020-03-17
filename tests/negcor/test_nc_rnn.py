from pyridge.experiment.check import check_algorithm


def test_nc_rnn():
    hyperparameter_nn = {
        'max_iter': [100],
        'activation': ['sigmoid'],
        'solver': ['irprop'],
        'batch_size': [25],
        'hidden_neurons': [5, 10],
        'learning_rate': [0.01],
        'size': [5],
        'lambda_': [0.01]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='NegativeCorrelationRNN',
                                 hyperparameter=hyperparameter_nn,
                                 metric_list=['accuracy', 'rmse'])


def test_nc_rnn_regression():
    hyperparameter_nn = {'max_iter': [100],
                         'activation': ['sigmoid'],
                         'hidden_neurons': [5, 10],
                         'learning_rate': [0.001],
                         'solver': ['irprop'],
                         'batch_size': [50],
                         'size': [5],
                         'lambda_': [0.001]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='NegativeCorrelationRNN',
                                 hyperparameter=hyperparameter_nn,
                                 metric_list=['rmse'],
                                 classification=False)
