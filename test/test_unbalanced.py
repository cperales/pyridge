from sklearn.datasets import make_classification
from pyridge.utils import metric_dict
from pyridge.algorithm import algorithm_dict
from pyridge.utils import cross_validation, j_encode
import logging

logger = logging.getLogger('PyRidge')
logger.setLevel(logging.DEBUG)


def test():
    # Data
    train_data, train_target =\
        make_classification(n_samples=500, n_features=2, n_informative=2,
                            n_redundant=0, n_repeated=0, n_classes=3,
                            n_clusters_per_class=1,
                            weights=[0.01, 0.05, 0.94],
                            class_sep=0.8, random_state=0)

    train_j_target = j_encode(train_target)

    # Algorithm
    metric = metric_dict['accuracy']
    algorithm = algorithm_dict['KRidge']
    C_range = [0.1]  # [10 ** i for i in range(-2, 3)]
    k_range = [0.1]  # [10 ** i for i in range(-2, 3)]
    kernel_fun = 'rbf'

    hyperparameters = {'kernelFun': kernel_fun,
                       'C': C_range,
                       'k': k_range}

    clf = algorithm()
    clf.set_cv_range(hyperparameters)
    cross_validation(classifier=clf, train_data=train_data, train_target=train_j_target)
    pred_targ = clf.predict(test_data=train_data)
    acc = metric(pred_targ=pred_targ,
                 real_targ=train_j_target)

    logger.info('Accuracy for algorithm KRidge and an unbalanced dataset,'
                ' is {}'.format(acc))


if __name__ == '__main__':
    test()
