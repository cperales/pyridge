from pyridge.experiment.check import check_algorithm


def test_bagging_stepwise():
    hyperparameter_bagging = {'activation': ['sigmoid'],
                              'reg': [10 ** i for i in range(-1, 2)],
                              'hidden_neurons': [10],
                              'size': [5]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='BaggingStepwiseELM',
                                 hyperparameter=hyperparameter_bagging,
                                 metric_list=['accuracy', 'rmse'])


def test_bagging_stepwise_regression():
    hyperparameter_bagging = {'activation': ['sigmoid'],
                              'reg': [10 ** i for i in range(-1, 2)],
                              'hidden_neurons': [10],
                              'size': [5]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='BaggingStepwiseELM',
                                 hyperparameter=hyperparameter_bagging,
                                 metric_list=['rmse'],
                                 classification=False)
