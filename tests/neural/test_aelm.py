from pyridge.experiment.test import test_algorithm


def test_adaboost():
    hyperparameter = {'activation': ['sigmoid'],
                      'reg': [10 ** i for i in range(-1, 2)],
                      'hidden_neurons': [10],
                      'size': [5]}

    value_dict = test_algorithm(folder='data',
                                dataset='iris',
                                algorithm='AdaBoostELM',
                                hyperparameter=hyperparameter,
                                metric_list=['accuracy'])
