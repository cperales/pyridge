from pyridge.experiment.test import test_algorithm


def test_bagging_stepwise():
    hyperparameter_bagging = {'activation': ['sigmoid'],
                              'reg': [10 ** i for i in range(-1, 2)],
                              'hidden_neurons': [10],
                              'size': [5]}
    algorithm = 'BaggingStepwiseELM'

    value_dict = test_algorithm(folder='data',
                                dataset='iris',
                                algorithm=algorithm,
                                hyperparameter=hyperparameter_bagging,
                                metric_list=['accuracy', 'rmse'])


def test_bagging_stepwise_regression():
    hyperparameter_bagging = {'activation': ['sigmoid'],
                              'reg': [10 ** i for i in range(-1, 2)],
                              'hidden_neurons': [10],
                              'size': [5]}
    algorithm = 'BaggingStepwiseELM'

    value_dict = test_algorithm(folder='data_regression',
                                dataset='housing',
                                algorithm=algorithm,
                                hyperparameter=hyperparameter_bagging,
                                metric_list=['rmse'],
                                classification=False)
