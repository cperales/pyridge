from pyridge.experiment.check import check_algorithm


def test_pcaelm():
    hyperparameter_elm = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'pca_perc': [0.9]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='PCAELM',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['accuracy', 'rmse'])


def test_pcaelm_regression():
    hyperparameter_elm = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'pca_perc': [0.9]}
    value_dict = check_algorithm(folder='data_regression',
                                 dataset='housing',
                                 algorithm='PCAELM',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['rmse'],
                                 classification=False)
