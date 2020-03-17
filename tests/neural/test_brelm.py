from pyridge.experiment.test import test_algorithm


def test_boosting_ridge():
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [10],
                            'size': [5]}
    value_dict = test_algorithm(folder='data',
                                dataset='iris',
                                algorithm='BoostingRidgeELM',
                                hyperparameter=hyperparameter_boost,
                                metric_list=['accuracy', 'rmse'])


def test_boosting_ridge_elm_regression():
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [10],
                            'size': [5]}
    value_dict = test_algorithm(folder='data_regression',
                                dataset='housing',
                                algorithm='BoostingRidgeELM',
                                hyperparameter=hyperparameter_boost,
                                metric_list=['rmse'],
                                classification=False)
