from .classifier import Classifier
from postprocess import loss
import numpy as np
import itertools
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class CVMethod(Classifier):

    C = 0
    ensemble_size = 1
    grid_param = {'C': C}

    def config(self, train):
        # Cross validation
        cv_param_names = list(self.grid_param.keys())
        list_comb = [self.grid_param[name] for name
                     in cv_param_names]

        # Init the CV criteria
        best_cv_criteria = np.inf
        n_folds = 5
        # test_size = len(train['data']) / n_folds
        kf = KFold(n_splits=n_folds)

        for current_comb in itertools.product(*list_comb):
            # L = np.zeros(n_folds)
            L = []
            clf_list = []

            for train_index, test_index in kf.split(train['data']):
                # print('Train:', b[train_index])
                # print('Test:', b[test_index])

                param = {cv_param_names[i]: current_comb[i]
                         for i in range(len(cv_param_names))}

                train_fold = {'data': train['data'][train_index],
                              'target': train['target'][train_index]}
                self.fit(train=train_fold, parameters=param)

                test_fold = train['data'][test_index]
                pred = self.predict(test_data=test_fold)

                clf_param = self.save_param()
                clf_list.append(clf_param)

                L.append(loss(train['target'][test_index], pred))

            L = np.array(L, dtype=np.float)
            current_cv_criteria = np.mean(L)

            if current_cv_criteria < best_cv_criteria:
                position = L.index(min(L))
                best_clf_param = clf_list[position]
                best_cv_criteria = current_cv_criteria

        # optimals = matrix_cell[]
        self.load_clf(best_clf_param)
