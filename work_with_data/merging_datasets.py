import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# # MERGING TRAINING AND SETS DATASETS .1 IN ONE FILE
data_folder = 'data'

pair_dir_files = []

for dir_name, subdir_name, file_names in os.walk(data_folder):
    valid_file_names = []
    for file_name in file_names:
        if '.0' == file_name[-2:]:  # Storing first n_fold
            valid_file_names.append(file_name)
        else:
            file_to_delete = os.path.join(dir_name, file_name)
            os.remove(file_to_delete)
    if len(valid_file_names) > 0:
        pair_dir_files.append((dir_name, valid_file_names))

# SPLITTING ONE DATASET FILE IN N_FOLDS
n_fold = 10
for pair_file in pair_dir_files:
    list_whole_file = []
    dir_name, valid_file_names = pair_file
    for file_name in valid_file_names:
        file = os.path.join(dir_name, file_name)
        df_file = pd.read_csv(file,
                              sep='\s+',
                              header=None)
        list_whole_file.append(df_file)
    if 'train_' in valid_file_names[0]:
        whole_file_name = valid_file_names[0][6:-2]
    else:
        whole_file_name = valid_file_names[0][5:-2]
    whole_file_name_path = os.path.join(dir_name, whole_file_name)
    whole_file = pd.concat(list_whole_file)
    whole_file.to_csv(whole_file_name_path + '_all',
                      sep=' ',
                      header=None,
                      index=False)

    skf = StratifiedKFold(n_splits=n_fold, shuffle=False)
    target_position = whole_file.columns[-1]
    x = whole_file[[i for i in range(target_position)]]
    y = whole_file[[target_position]]
    # Shuffle false in order to preserv
    i = 0
    for train_index, test_index in skf.split(X=x, y=y):
        x_train_fold = x.iloc[train_index]
        y_train_fold = y.iloc[train_index]
        train_fold = pd.concat([x_train_fold, y_train_fold], axis=1)
        train_fold_name = 'train_' + whole_file_name + '.' + str(i)
        train_fold_name_path = os.path.join(dir_name, train_fold_name)
        train_fold.to_csv(train_fold_name_path,
                          sep=' ',
                          header=None,
                          index=False)

        x_test_fold = x.iloc[test_index]
        y_test_fold = y.iloc[test_index]
        test_fold = pd.concat([x_test_fold, y_test_fold], axis=1)
        test_fold_name = 'test_' + whole_file_name + '.' + str(i)
        test_fold_name_path = os.path.join(dir_name, test_fold_name)
        test_fold.to_csv(test_fold_name_path,
                         sep=' ',
                         header=None,
                         index=False)

        i += 1
