from pyridge.experiment.check import check_algorithm


def test_pcaldaelm():
    hyperparameter_elm = {'activation': ['sigmoid'],
                          'reg': [10 ** i for i in range(-1, 2)],
                          'pca_perc': [0.9]}
    value_dict = check_algorithm(folder='data',
                                 dataset='iris',
                                 algorithm='PCALDAELM',
                                 hyperparameter=hyperparameter_elm,
                                 metric_list=['accuracy'])
