from algorithm import *
from postprocess.result_metric import accuracy
from utility.target_encode import j_encode
import json
import pandas as pd
import logging
import os
logging.basicConfig(level=logging.DEBUG)

algorithm_dict = {'NELM': NELM}
metric_dict = {'accuracy': accuracy}

# Reading JSON
with open('config/NELM_newthyroid.json', 'r') as cfg:
    config_options = json.load(cfg)

# Training data and target

training_file_name = os.path.join(config_options['Data']['folder'],
                                  config_options['Data']['trainingDataset'])
training_file = pd.read_csv(training_file_name,
                            sep='\s+',
                            header=None)
training_file_matrix = training_file.as_matrix()
training_file_matrix_t = training_file_matrix.transpose()
training_target = training_file_matrix_t[0].transpose()
training_data = training_file_matrix_t[1:].transpose()

# Testing data and target
testing_file_name = os.path.join(config_options['Data']['folder'],
                                 config_options['Data']['testingDataset'])
testing_file = pd.read_csv(testing_file_name,
                           sep='\s+',
                           header=None)
testing_file_matrix = testing_file.as_matrix()
testing_file_matrix_t = testing_file_matrix.transpose()
testing_target = testing_file_matrix_t[0].transpose()
testing_data = testing_file_matrix_t[1:].transpose()

# Reading parameters
hyperparameters = config_options['Algorithm']['hyperparameters']

# Instancing classifier
# clf = algorithm_dict[config_options['Algorithm']['name']](hyperparameters)
clf = algorithm_dict[config_options['Algorithm']['name']]()
clf.set_conf(hyperparameters)

# Fitting classifier
# For testing
parameters = {'C': 0, 'hidden_neurons': 50}
training_J_target = j_encode(training_target)
clf.fit(train={'data': training_data, 'target': training_target}, parameters=parameters)

# Running test
predicted_labels = clf.classify(data=testing_data)

# Metrics
metric_value_dict = {}
for metric in config_options['Report']['metrics']:
    metric_function = metric_dict[metric.lower()]
    metric_value = metric_function(predicted_targets=predicted_labels,
                                   real_targets=testing_target)
    metric_value_dict.update({metric: metric_value})
    logging.debug('{} = {}'.format(metric, metric_value))
# acc = accuracy(predicted_targets=predicted_labels,
#                real_targets=testing_target)

# Report
report = {
    'Training dataset': training_file_name,
    'Testing dataset': testing_file_name,
    'Classifier': config_options['Algorithm']['name']}

# Hyperparameters added
report.update(hyperparameters)

# Metrics added
report.update(metric_value_dict)

df_report = pd.DataFrame(report, index=[0])
df_report.to_csv(config_options['Report']['folder'] +
                 config_options['Report']['report_name'] +
                 '.csv',
                 sep=';')

