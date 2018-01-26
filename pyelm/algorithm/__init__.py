from .nelm import NELM
from .adaboost import AdaBoostNELM
from .kelm import KELM

algorithm_dict = {'NELM': NELM,
                  'AdaBoostNELM': AdaBoostNELM,
                  'KELM': KELM}
