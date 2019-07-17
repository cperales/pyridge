from pyridge.experiment.test import test_algorithm


def test_diverse_elm():
    hyperparameter_div = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'div': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10],
                          'size': [5]}
    algorithm = 'DiverseELM'

    test_algorithm(folder='data',
                   dataset='iris',
                   algorithm=algorithm,
                   hyperparameter=hyperparameter_div,
                   metric_list=['accuracy', 'rmse', 'diversity'])


def test_diverse_elm_regression():
    hyperparameter_div = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'div': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10],
                          'size': [5]}
    algorithm = 'DiverseELM'

    test_algorithm(folder='data_regression',
                   dataset='housing',
                   algorithm=algorithm,
                   hyperparameter=hyperparameter_div,
                   metric_list=['rmse', 'diversity'],
                   classification=False)
