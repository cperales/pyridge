import json
import os
from time import perf_counter

import pandas as pd
from sklearn import preprocessing
import logging

from pyelm.algorithm import *
from pyelm.clf_utility import save_classifier
from pyelm.clf_utility.cross_val import *
from pyelm.clf_utility.metric import accuracy
from pyelm.clf_utility.target_encode import j_encode

logger_pyelm = logging.getLogger('PyELM')
logger_pyelm.setLevel(logging.DEBUG)

algorithm_dict = {'KELM': KELM}
metric_dict = {'accuracy': accuracy}

# Reading JSON
with open('config/KELM_newthyroid.json', 'r') as cfg:
    config_options = json.load(cfg)

# Training data and target

training_file_name = os.path.join(config_options['Data']['folder'],
                                  config_options['Data']['trainingDataset'])
training_file = pd.read_csv(training_file_name,
                            sep='\s+',
                            header=None)
training_file_matrix = training_file.as_matrix()
training_file_matrix_t = training_file_matrix.transpose()
training_target = training_file_matrix_t[-1].transpose()
training_data = training_file_matrix_t[:-1].transpose()

# Testing data and target
testing_file_name = os.path.join(config_options['Data']['folder'],
                                 config_options['Data']['testingDataset'])
testing_file = pd.read_csv(testing_file_name,
                           sep='\s+',
                           header=None)
testing_file_matrix = testing_file.as_matrix()
testing_file_matrix_t = testing_file_matrix.transpose()
testing_target = testing_file_matrix_t[-1].transpose()
testing_data = testing_file_matrix_t[:-1].transpose()

training_data = preprocessing.scale(training_data)
testing_data = preprocessing.scale(testing_data)

# Reading parameters
hyperparameters = config_options['Algorithm']['hyperparameters']

# Instancing classifier
# clf = algorithm_dict[config_options['Algorithm']['name']](hyperparameters)
clf = algorithm_dict[config_options['Algorithm']['name']]()

# cross_validation(clf, hyperparameters)

clf.set_range_param(hyperparameters)
training_J_target = j_encode(training_target)
n_targ = training_J_target.shape[1]
testing_j_target = j_encode(testing_target, n_targ=n_targ)

train_dict = {'data': training_data, 'target': training_J_target}

# # Fitting classifier
# Profiling
from cProfile import Profile
prof = Profile()
prof.enable()
time_1 = perf_counter()


n_run = 10
acc = 0
for i in range(n_run):
    cross_validation(classifier=clf, train=train_dict)
    predicted_labels = clf.predict(test_data=testing_data)
    acc += accuracy(predicted_targets=predicted_labels,
                    real_targets=testing_j_target)
acc = acc / n_run

# Saving classifier
save_classifier(clf, 'ELM_newthyroid.clf')

# Profiling
time_2 = perf_counter()
prof.disable()  # don't profile the generation of stats

try:
    prof.dump_stats('profile/mystats.prof')
except FileNotFoundError:  # There is no 'profile' folder
    pass

logger_pyelm.debug('{} seconds elapsed'.format(time_2 - time_1))

logger_pyelm.info('Average accuracy in {} iterations, algorithm {} and dataset {} is {}'.format(n_run,
                                                                                          config_options['Algorithm']['name'],
                                                                                          config_options['Data']['trainingDataset'],
                                                                                          acc))

# # Running different metrics
# predicted_labels = clf.predict(test_data=testing_data)
# n_targ = predicted_labels.shape[1]
# testing_j_target = j_encode(testing_target, n_targ=n_targ)
# # Metrics
# metric_value_dict = {}
# for metric in config_options['Report']['metrics']:
#     metric_function = metric_dict[metric.lower()]
#     metric_value = metric_function(predicted_targets=predicted_labels,
#                                    real_targets=testing_j_target)
#     metric_value_dict.update({metric: metric_value})
#     logger.info('{} = {}'.format(metric, metric_value))
# acc = accuracy(predicted_targets=predicted_labels,
#                real_targets=testing_target)

# # Report
# report = {
#     'Training dataset': training_file_name,
#     'Testing dataset': testing_file_name,
#     'Classifier': config_options['Algorithm']['name']}
#
# # Hyperparameters added
# report.update(hyperparameters)
#
# # Metrics added
# report.update(metric_value_dict)
#
# df_report = pd.DataFrame(report, index=[0])
# df_report.to_csv(config_options['Report']['folder'] +
#                  config_options['Report']['report_name'] +
#                  '.csv',
#                  sep=';')

