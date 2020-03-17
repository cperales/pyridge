from pyridge.experiment.check import check_algorithm


def test_ncelm():
    hyperparameter_inc = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-2, 3)],
                          'lambda_': [10 ** i for i in range(-4, -2)],
                          'max_iter_': [5],
                          'hidden_neurons': [10],
                          'size': [5]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='NegativeCorrelationELM',
                                 hyperparameter=hyperparameter_inc,
                                 metric_list=['accuracy', 'rmse'])


def test_ncelm_regression():
    hyperparameter_inc = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'lambda_': [10 ** i for i in range(-2, -1)],
                          'max_iter_': [5],
                          'hidden_neurons': [10],
                          'size': [5]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='NegativeCorrelationELM',
                                 hyperparameter=hyperparameter_inc,
                                 metric_list=['rmse'],
                                 classification=False)
