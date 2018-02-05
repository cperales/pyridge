from pyelm.utils import metric_dict
from pyelm.algorithm import algorithm_dict
from pyelm.utils import cross_validation
from pyelm.utils.preprocess import prepare_data

# Data
folder = 'data/newthyroid'
train_dataset = 'train_newthyroid.0'
train_data, train_j_target = prepare_data(folder=folder,
                                          dataset=train_dataset)
test_dataset = 'test_newthyroid.0'
n_targ = train_j_target.shape[1]

test_data, test_j_target = prepare_data(folder=folder,
                                        dataset=test_dataset,
                                        n_targ=n_targ)

# Algorithm
metric = metric_dict['accuracy']
algorithm = algorithm_dict['KELM']
C_range = [10**i for i in range(-2, 3)]
k_range = [10**i for i in range(-2, 3)]
kernel_fun = 'rbf'

hyperparameters = {'kernelFun': kernel_fun,
                   'C': C_range,
                   'k': k_range}

clf = algorithm()
clf.set_cv_range(hyperparameters)
cross_validation(classifier=clf, train_data=train_data, train_target=train_j_target)
pred_targ = clf.predict(test_data=test_data)
acc = metric(pred_targ=pred_targ,
             real_targ=test_j_target)

print('Finished!')