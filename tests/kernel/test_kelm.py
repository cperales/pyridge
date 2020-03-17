from pyridge.experiment.check import check_algorithm


def test_kelm():
    hyperparameter_kelm = {'kernel': ['rbf', 'linear'],
                           'reg': [10 ** i for i in range(-1, 2)],
                           'gamma': [10]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='KernelELM',
                                 hyperparameter=hyperparameter_kelm,
                                 metric_list=['accuracy', 'rmse'])


def test_kelm_regression():
    hyperparameter_kelm = {'kernel': ['rbf', 'linear'],
                           'reg': [10 ** i for i in range(-1, 2)],
                           'gamma': [10]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='KernelELM',
                                 hyperparameter=hyperparameter_kelm,
                                 metric_list=['rmse'],
                                 classification=False)
