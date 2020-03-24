from pyridge.experiment.check import check_algorithm
import pytest


def test_ancelm():
    hyperparameter_anc = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10],
                          'lambda_': [0.5, 1.0, 5.0],
                          'size': [5]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='AdaBoostNCELM',
                                 hyperparameter=hyperparameter_anc,
                                 metric_list=['accuracy'])


def test_ancelm_regression():
    hyperparameter_anc = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10],
                          'lambda_': [0.5, 1.0, 5.0],
                          'size': [5]}
    with pytest.raises(ValueError):
        value_dict = check_algorithm(folder='data_regression',
                                     dataset='housing',
                                     algorithm='AdaBoostNCELM',
                                     hyperparameter=hyperparameter_anc,
                                     metric_list=['rmse'],
                                     classification=False)
