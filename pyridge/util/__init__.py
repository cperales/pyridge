from .save_load import save_classifier, load_classifier
from .metric import *
from .target_encode import j_decode, j_encode, j_renorm
from .cross import train_predictor, cross_validation
from .preprocess import prepare_data
from .inverse import sp_solve as solver

# from .inverse import sp_solve as main_solver
# import numpy as np
#
# def solver(a, b):
#     try:
#         return main_solver(a=a, b=b)
#     except np.linalg.LinAlgError:
#         return np.linalg.solve(a=a, b=b)
