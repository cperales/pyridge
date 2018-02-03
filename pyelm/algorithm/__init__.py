from .nelm import NELM
from .adaboost import AdaBoostNELM
from .kelm import KELM
from .sklearn_svm import SklearnSVC

algorithm_dict = {'NELM': NELM,
                  'AdaBoostNELM': AdaBoostNELM,
                  'KELM': KELM,
                  'sklearnSVC': SklearnSVC}
