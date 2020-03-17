from pyridge.experiment.test import test_algorithm


def test_adaboost_nc():
    hyperparameter_anc = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'hidden_neurons': [10],
                          'lambda_': [0.5, 1.0, 5.0],
                          'size': [5]}
    value_dict = test_algorithm(folder='data',
                                dataset='iris',
                                algorithm='AdaBoostNCELM',
                                hyperparameter=hyperparameter_anc,
                                metric_list=['accuracy'])
