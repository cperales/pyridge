from pyridge.experiment.check import check_algorithm


def test_sobol():
    hyperparameter_elm = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='SobolELM',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['accuracy', 'rmse'])


def test_sobol_regression():
    hyperparameter_elm = {'activation': ['sigmoid'],
                          'reg': [0.001],
                          'hidden_neurons': [20]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='SobolELM',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['rmse'],
                                 classification=False)
