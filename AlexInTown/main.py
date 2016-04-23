# -*- coding: utf-8 -*-

import itertools
import os
#import json
import simplejson as json

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

# util
from utils.config_utils import *
from utils.submit_utils import *
from utils.json_utils import *
from utils.meta_utils import *

# preprocess
from preprocess import feat_transform
from preprocess import feat_filter

if __name__ == "__main__":

    stack_setting = {'0-Level':{'folder':'input',
                                'raw':{'train':'adult.data_train.csv',
                                       'test':'adult.data_test.csv'},
                                'filter':{'train':'filtered_train.csv',
                                          'test':'filtered_test.csv'},
                                'raw_extend':{'train':'raw_extend_train.csv',
                                              'test':'raw_extend_test.csv'},
                                'scaled':{'train':'scaled_train.csv',
                                          'test':'scaled_test.csv'},
                                'scaled_extend':{'train':'scaled_extend_train.csv',
                                                 'test':'scaled_extend_test.csv'},
                                'standard':{'train':'standard_train.csv',
                                            'test':'standard_test.csv'},
                                'standard_extend':{'train':'standard_extend_train.csv',
                                                   'test':'standard_extend_test.csv'},
                                'train':'train.csv',
                                'test':'test.csv'},
                     '1-Level':{'knn':{'meta_feature':{'folder':'output',
                                                       'train':'meta_knn_train.csv',
                                                       'test':'meta_knn_test.csv',
                                                       'header':'meta_knn,label'},
                                       'best_parameter':None,
                                       'cv':{'metrics':'auc',
                                             'folder':'output',
                                             'cv_out':'sk-knn-grid-scores.pkl', 
                                             'cv_pred_out':'sk-knn-grid-preds.pkl', 
                                             'refit_pred_out':'sk-knn-refit-preds.pkl'},
                                       'graph':{'folder':'graph',
                                                'name':'sk-knn-grid.png'}},
                                'gbdt_linear':{'upper':{'metrics':'f1',
                                                        'best_parameter':None,
                                                        'gbdt':{'folder':'model',
                                                                'train':'sk-gbdt_linear-grid_upper-train_model.pkl',
                                                                'test':'sk-gbdt_linear-grid_upper-test_model.pkl'},
                                                        'graph':{'folder':'graph',
                                                                 'name':'sk-gbdt_linear_upper-grid.png'}},
                                               'lower':{'meta_feature':{'folder':'output',
                                                                        'train':'meta_gbdt_linear_train.csv',
                                                                        'test':'meta_gbdt_linear_test.csv',
                                                                        'header':'meta_gbdt_linear,label'},
                                                        'best_parameter':None,
                                                        'cv':{'metrics':'auc',
                                                              'folder':'output',
                                                              'cv_out':'sk-gbdt_linear_lower-grid-scores.pkl', 
                                                              'cv_pred_out':'sk-gbdt_linear_lower-grid-preds.pkl', 
                                                              'refit_pred_out':'sk-gbdt_linear_lower-refit-preds.pkl'},
                                                        'graph':{'folder':'graph',
                                                                 'name':'sk-gbdt_linear_lower-grid.png'}}},
                                'rf':{'meta_feature':{'folder':'output',
                                                      'train':'meta_rf_train.csv',
                                                      'test':'meta_rf_test.csv',
                                                      'header':'meta_rf,label'},
                                      'best_parameter':None,
                                      'cv':{'metrics':'auc',
                                            'folder':'output',
                                            'cv_out':'sk-rf-grid-scores.pkl', 
                                            'cv_pred_out':'sk-rf-grid-preds.pkl', 
                                            'refit_pred_out':'sk-rf-refit-preds.pkl'},
                                      'graph':{'folder':'graph',
                                               'name':'sk-rf-grid.png'}},
                                'xgb':{'meta_feature':{'folder':'output',
                                                       'train':'meta_xgb_train.csv',
                                                       'test':'meta_xgb_test.csv',
                                                       'header':'meta_xgb,label'},
                                       'best_parameter':None,
                                       'cv':{'metrics':'auc',
                                             'folder':'output',
                                             'cv_out':'sk-xgb-grid-scores.pkl', 
                                             'cv_pred_out':'sk-xgb-grid-preds.pkl', 
                                             'refit_pred_out':'sk-xgb-refit-preds.pkl'},
                                       'graph':{'folder':'graph',
                                                'name':'sk-xgb-grid.png'}},
                                'meta_features':{'folder':'output',
                                                 'train':'meta_train.csv',
                                                 'test':'meta_test.csv'}},
                     '2-Level':{'ridge':{'best_parameter':None,
                                         'cv':{'metrics':'auc',
                                               'folder':'output',
                                               'cv_out':'sk-ridge-grid-scores.pkl', 
                                               'cv_pred_out':'sk-ridge-grid-preds.pkl', 
                                               'refit_pred_out':'sk-ridge-refit-preds.pkl'},
                                         'graph':{'folder':'graph',
                                                  'name':'blending_ridge.png'}},
                                'blending':{'folder':'output',
                                            'train':'meta_train_at_blend.csv',
                                            'ptrain':'meta_ptrain_at_blend.csv',
                                            'ptest':'meta_ptest_at_blend.csv',
                                            'hold_out_ratio':0.2,
                                            'model':'sk-ridge-blend.pkl.gz',
                                            'weight':'meta_weight.csv'}},
                     'setting':{'folder':'stack_setting',
                                'name':'stack_setting.json'}
                     }


    # 1. 0-Level
    # 1.1 feature filtering
    feat_filter.main(stack_setting_ = stack_setting)


    # 2. 1-Level
    # experiment as 1-Level(stacking)
    from experiment.model_wrappers import *
    from experiment.param_search  import *


    # kNN
    from experiment.stacking.run_knn_grid_search import *
    param = {'model_type':[KNeighborsClassifier],
             'n_neighbors':[1, 2, 4, 8, 16, 24, 32, 64], #[1, 2, 4, 8, 16, 24, 32, 64]
             'weights':['uniform', 'distance'],
             'algorithm':['ball_tree'], 
             'leaf_size':[30], 
             'metric':['minkowski'], 
             'p':[2], 
             'n_jobs':[4]}
    knn_best_params, knn_best_score = knn_grid_search(stack_setting_ = stack_setting,
                                                      param_keys = param.keys(), 
                                                      param_vals = param.values(),
                                                      num_proc = 1)
    stack_setting['1-Level']['knn']['best_parameter'] = knn_best_params


    # GBDT + Linear Classifier(LR)
    from experiment.stacking.run_gbdt_plus_liner_classifier_grid_search import *
    model_folder = stack_setting['1-Level']['gbdt_linear']['upper']['gbdt']['folder']
    model_train_fname = stack_setting['1-Level']['gbdt_linear']['upper']['gbdt']['train']
    model_train_fname = os.path.join(Config.get_string('data.path'), 
                                     model_folder, 
                                     model_train_fname)
    model_folder = stack_setting['1-Level']['gbdt_linear']['upper']['gbdt']['folder']
    model_test_fname = stack_setting['1-Level']['gbdt_linear']['upper']['gbdt']['test']
    model_test_fname = os.path.join(Config.get_string('data.path'), 
                                    model_folder, 
                                    model_test_fname)
    if os.path.isfile(model_train_fname):
        os.remove(model_train_fname)
    if os.path.isfile(model_test_fname):
        os.remove(model_test_fname)
    upper_param = {'model_type':[GradientBoostingClassifier],
                   'n_estimators': [10, 100, 500, 1000], #[10, 100, 500]
                   'loss': ['deviance'], 
                   'random_state': [0], 
                   'subsample': [0.1, 0.5, 0.9], #[0.1, 0.5, 0.9]
                   'max_features': [5,10], #[5, 10]
                   'max_leaf_nodes': [5, 10, 15, 20], #[5, 10, 15, 20]
                   'learning_rate': [0.01, 0.1, 1.0], #[0.01, 0.1, 1.0]
                   'max_depth': [2, 4, 6, 8], #[2, 4, 6, 8]
                   'min_samples_leaf': [2, 4, 6, 8]}#[2, 4, 6, 8]
    lower_best_params = {'LR-L1':None, 'LR-L2':None, 'SVM-L1':None, 'SVM-L2':None}

    # LR L2
    lower_param = {'model_type':[LogisticRegression],
                   'C': [0.01, 0.1, 1.0, 10.0, 100.0],#[0.01, 0.1, 1.0, 10.0, 100.0]
                   'penalty' : ['l2'],#['l1','l2']
                   'fit_intercept':[True]}

    upper_best_param, lower_best_param = gbdt_plus_liner_classifier_grid_search(
        stack_setting_ = stack_setting,
        upper_param_keys=upper_param.keys(), 
        upper_param_vals=upper_param.values(),
        lower_param_keys=lower_param.keys(), 
        lower_param_vals=lower_param.values()
        )
    stack_setting['1-Level']['gbdt_linear']['upper']['best_parameter'] = upper_best_param
    lower_best_params['LR-L2'] = lower_best_param


    # LR L1
    lower_param = {'model_type':[LogisticRegression],
                   'C': [0.01, 0.1, 1.0, 10.0, 100.0],#[0.01, 0.1, 1.0, 10.0, 100.0]
                   'penalty' : ['l1'],#['l1','l2']
                   'fit_intercept':[True]}

    upper_best_param, lower_best_param = gbdt_plus_liner_classifier_grid_search(
        stack_setting_ = stack_setting,
        upper_param_keys=upper_param.keys(), 
        upper_param_vals=upper_param.values(),
        lower_param_keys=lower_param.keys(), 
        lower_param_vals=lower_param.values()
        )
    stack_setting['1-Level']['gbdt_linear']['upper']['best_parameter'] = upper_best_param
    lower_best_params['LR-L1'] = lower_best_param

    # SVM L2
    lower_param = {'model_type':[LinearSVC],
                   'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                   'loss':['hinge'],
                   'penalty':['l2'],
                   'tol':[1.0e-3],
                   'random_state':[0],
                   'fit_intercept':[True]}

    upper_best_param, lower_best_param = gbdt_plus_liner_classifier_grid_search(
        stack_setting_ = stack_setting,
        upper_param_keys=upper_param.keys(), 
        upper_param_vals=upper_param.values(),
        lower_param_keys=lower_param.keys(), 
        lower_param_vals=lower_param.values()
        )
    #stack_setting['1-Level']['gbdt_linear']['upper']['best_parameter'] = upper_best_param
    lower_best_params['SVM-L2'] = lower_best_param

    # SVM L1
    lower_param = {'model_type':[LinearSVC],
                   'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                   'loss':['squared_hinge'],
                   'penalty':['l1'],
                   'dual':[False],
                   'tol':[1.0e-3],
                   'random_state':[0],
                   'fit_intercept':[True]}

    stack_setting['1-Level']['gbdt_linear']['lower']['best_parameter'] = lower_best_params
    upper_best_param, lower_best_param = gbdt_plus_liner_classifier_grid_search(
        stack_setting_ = stack_setting,
        upper_param_keys=upper_param.keys(), 
        upper_param_vals=upper_param.values(),
        lower_param_keys=lower_param.keys(), 
        lower_param_vals=lower_param.values()
        )
    lower_best_params['SVM-L1'] = lower_best_param


    ## neural-net
    #from experiment.run_neural_network import *
    #neural_net = make_nn_instance()
    #nn_fit(neural_net)


    # random forest
    from experiment.stacking.run_rand_forest_grid_search import *
    param = {'model_type':[RandomForestClassifier],
             'max_depth':[2,4,8],#[2,4,8]
             'min_samples_leaf':[3],
             'max_leaf_nodes':[5, 7, 9],#[5, 7, 9]
             'criterion':['entropy', 'gini'],#['entropy', 'gini']
             'n_estimators':[50, 100, 500, 1000],#[50, 100, 500]
             'n_jobs':[5]}

    rf_best_params, rf_best_score = rf_grid_search(stack_setting_ = stack_setting,
                                                   param_keys = param.keys(), 
                                                   param_vals = param.values(),
                                                   num_proc = 1)
    stack_setting['1-Level']['rf']['best_parameter'] = rf_best_params


    # xgboost
    from hyperopt import hp
    from xgboost import XGBClassifier
    from experiment.stacking.run_xgb_param_search import *
    exp = ExperimentL1(data_folder = stack_setting['0-Level']['folder'],
                       train_fname = stack_setting['0-Level']['train'],
                       test_fname = stack_setting['0-Level']['test'])
    param = {'model_type':[XGBClassifier], 
             'max_depth':[4, 8], #[4, 8]
             'min_child_weight':[3, 6], #[3, 6]
             'subsample':[0.1, 0.5, 0.9], #[0.1, 0.5, 0.9]
             'colsample_bytree':[0.1, 0.5, 0.9],#[0.1, 0.5, 0.9]
             'learning_rate':[0.01, 0.1, 1.0], #[0.01, 0.1, 1.0]
             'silent':[1], 
             'objective':['binary:logistic'], 
             'nthread':[6],
             'n_estimators':[50, 100, 500, 1000], #[50, 100, 300, 500]
             'seed':[9438]}
    xgb_best_params, xgb_best_score = xgb_grid_search(stack_setting_ = stack_setting,
                                                      exp = exp, 
                                                      param_keys = param.keys(), 
                                                      param_vals = param.values(),
                                                      num_proc = 6)
    # hyper opt : bayesian optimizaition
    #xgb_best_params, xgb_best_score = xgb_bayes_search(exp)
    #xgb_submmision(exp, xgb_best_params)
    stack_setting['1-Level']['xgb']['best_parameter'] = xgb_best_params


    # Combine Meta Feature
    combine_meta_features(stack_setting_ = stack_setting)
    make_hold_out(stack_setting_ = stack_setting)

    
    # 3. blending
    # 3.1 calculate blending weight at 2-level throught hold out set
    from experiment.model_wrappers import *
    from experiment.blending.run_ridge_grid_search import *

    param = {'model_type':[RidgeClassifier],
             'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
             'max_iter':[100],
             'fit_intercept':[True],
             'solver':['auto', 'sag'],
             'random_state' : [1]}

    ridge_best_params, ridge_best_score = ridge_grid_search(stack_setting_ = stack_setting,
                                                            param_keys = param.keys(), 
                                                            param_vals = param.values(),
                                                            num_proc = 1)
    stack_setting['2-Level']['ridge']['best_parameter'] = ridge_best_params

    # 3.2 blending
    accuracy, precision, recall, f = ridge_blend(stack_setting_ = stack_setting,
                                                 best_param_ = ridge_best_params)
    print "accuracy=%1.5e, precision=%1.5e, recall=%1.5e, f=%1.5e" % (accuracy, precision, recall, f)


    # 4. dump stack setting
    dump_stacking_setting(stack_setting_ = stack_setting)
    
