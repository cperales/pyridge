import os
import json
import copy
from template import json_templates

# # Folder to search the data
data_folder = 'data'  # Try/except something like "is folder"
config_folder = 'config'  # Try/except something like "is folder"
pyelm_tests = ['hepatitis', 'newthyroid']

# List of the pair directory / data
pair_dir_files = []

values = [str(i) for i in range(10)]

for dirname, subdirname, file_names in os.walk(data_folder):
    valid_file_names = []
    for filename in file_names:
        if filename[-1:] in values:
                valid_file_names.append(filename)
    if len(valid_file_names) > 0:
        pair_dir_files.append((dirname, valid_file_names))

algorithms = ['AdaBoostNELM',
              'NELM',
              'KELM']

# All the train-test file names are stored, along their directory name.
for algorithm in algorithms:
    algorithm_tuple = json_templates[algorithm]
    config_name, dict_template = algorithm_tuple

    # A list of tests in a JSON
    list_of_experiments = []

    for dirname, file_names in pair_dir_files:
        each_dict = copy.deepcopy(dict_template)

        list_train_file_names = [file_name for file_name in file_names
                                 if 'train' in file_name]

        list_test_file_names = [file_name for file_name in file_names
                                if 'test' in file_name]

        name_dataset = list_train_file_names[0][:-2]
        name_dataset = name_dataset.replace('train_', '')

        dataset = []
        for train_file in list_train_file_names:
            number_train_file = train_file[-1]
            test_file = train_file  # If no test file has been found
            
            for t in list_test_file_names:
                if number_train_file in t:
                    test_file = t
                    list_test_file_names.remove(t)
                    break

            dataset.append([train_file, test_file])

        # if dataset in pyelm_tests:
        each_dict["Data"]["dataset"] = dataset
        each_dict["Data"]["folder"] = dirname
        each_dict["Report"]["report_name"] = name_dataset

        list_of_experiments.append(each_dict)

    with open(os.path.join(config_folder, config_name), 'w') as outfile:
        json.dump(list_of_experiments, outfile)

# # A JSON by each excercise
# for dirname, file_names in pair_dir_files:
#     each_dict = copy.deepcopy(dict_template)
    
#     if len(file_names) > 2:
#         raise ValueError('It is expected a couple train-test in each dir')
    
#     trainingDataset = file_names[0] if 'train' in file_names[0] else file_names[1]
#     testingDataset = file_names[1] if 'test' in file_names[1] else file_names[0]
#     dataset = trainingDataset.replace('train', '')
#     dataset = dataset.replace('_', '')
#     dataset = dataset.replace('.1', '')
#     if dataset in pyelm_tests:
#         each_dict["Data"]["trainingDataset"] = trainingDataset
#         each_dict["Data"]["testingDataset"] = testingDataset
#         each_dict["Data"]["folder"] = dirname
#         each_dict["Report"]["report_name"] = 'experiment_' + dataset
#         json_name = dataset + '.json'

#         with open(os.path.join(config_folder, json_name), 'w') as outfile:
#             json.dump(each_dict, outfile)
