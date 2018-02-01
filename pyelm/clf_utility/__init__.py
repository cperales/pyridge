from .save_load import save_classifier, load_classifier
from .metric import accuracy, loss
from .target_encode import *
from .cross_val import cross_validation

metric_dict = {'accuracy': accuracy}
