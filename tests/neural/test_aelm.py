from pyridge.experiment.check import check_algorithm
import pytest


def test_adaboost():
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [10],
                            'size': [5]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='AdaBoostELM',
                                 hyperparameter=hyperparameter_boost,
                                 metric_list=['accuracy'])


def test_adaboost_regression():
    hyperparameter_boost = {'activation': ['sigmoid'],
                            'reg': [10 ** i for i in range(-1, 2)],
                            'hidden_neurons': [10],
                            'size': [5]}
    with pytest.raises(ValueError):
        value_dict = check_algorithm(folder='data_regression',
                                     dataset='housing',
                                     algorithm='AdaBoostELM',
                                     hyperparameter=hyperparameter_boost,
                                     metric_list=['rmse'],
                                     classification=False)
