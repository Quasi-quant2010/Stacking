# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
from sklearn.neighbors import KNeighborsClassifier
from experiment.stacking.experiment_l1 import ExperimentL1
from experiment.param_search import GridSearch
from experiment.model_wrappers import SklearnModel
from utils.config_utils import Config

def knn_grid_search(stack_setting_,
                    param_keys=None, param_vals=None,
                    num_proc=None):

    if stack_setting_ is None:
        sys.stderr.write('You have no setting Json file\n')
        sys.exit()

    if num_proc is None:
        num_proc = 6

    if param_keys is None:
        param_keys = ['model_type', 'n_neighbors', 'weights',
                      'algorithm', 'leaf_size', 'metric', 'p', 'n_jobs']

    if param_vals is None:
        param_vals = [[KNeighborsClassifier], [1, 2, 4, 8, 16, 24, 32, 64], ['uniform', 'distance'],
                      ['ball_tree'], [30], ['minkowski'], [2], [4]]


    exp = ExperimentL1(data_folder = stack_setting_['0-Level']['folder'],
                       train_fname = stack_setting_['0-Level']['train'], 
                       test_fname = stack_setting_['0-Level']['test'])

    gs = GridSearch(SklearnModel, exp, param_keys, param_vals,
                    cv_folder = stack_setting_['1-Level']['knn']['cv']['folder'],
                    cv_out = stack_setting_['1-Level']['knn']['cv']['cv_out'], 
                    cv_pred_out = stack_setting_['1-Level']['knn']['cv']['cv_pred_out'], 
                    refit_pred_out = stack_setting_['1-Level']['knn']['cv']['refit_pred_out'])
    best_param, best_score = gs.search_by_cv(validation_metrics = stack_setting_['1-Level']['knn']['cv']['metrics'])

    # get meta_feature
    exp.write2csv_meta_feature(
        model = KNeighborsClassifier(),
        meta_folder = stack_setting_['1-Level']['knn']['meta_feature']['folder'],
        meta_train_fname = stack_setting_['1-Level']['knn']['meta_feature']['train'],
        meta_test_fname = stack_setting_['1-Level']['knn']['meta_feature']['test'],
        meta_header = stack_setting_['1-Level']['knn']['meta_feature']['header'],
        best_param_ = best_param
        )

    return best_param, best_score
