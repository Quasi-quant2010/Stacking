# -*- coding:utf-8 -*-

import os
from experiment.stacking.experiment_l1 import ExperimentL1
from utils.config_utils import Config
from utils.submit_utils import save_submissions
from experiment.model_wrappers import *
from xgboost import XGBClassifier
import experiment.param_search as param_search
from hyperopt import hp

from sklearn.base import BaseEstimator, TransformerMixin, clone

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pandas as pd
import numpy as np

def xgb_bayes_search(exp, 
                     param_keys=None, param_vals=None,
                     num_proc=None):
    if num_proc is None:
        num_proc = 4

    if param_keys is None:
        param_keys = ['model_type', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
                      'learning_rate', 'silent', 'objective', 'nthread', 'n_estimators', 'seed']
    if param_vals is None:
        param_space = {'model_type': XGBClassifier, 'max_depth': hp.quniform('max_depth', 6, 9, 1),
                       'min_child_weight': hp.quniform('min_child_weight', 3, 7, 1),
                       'subsample': hp.uniform('subsample', 0.5, 1.0),
                       'colsample_bytree': hp.uniform('colsample', 0.5, 1.0),
                       'learning_rate': hp.uniform('eta', 0.01, 0.02),
                       'silent': 1, 'objective': 'binary:logistic',
                       'nthread': num_proc, 'n_estimators': 400, 'seed': 9438}
    
    
    bs = param_search.BayesSearch(SklearnModel, exp, param_keys, param_space,
                                  cv_out='xgb-bayes-scores.pkl',
                                  cv_pred_out='xgb-bayes-preds.pkl')
    best_param, best_score = bs.search_by_cv()
    param_search.write_cv_res_csv(cv_out = 'xgb-bayes-scores.pkl', 
                                  cv_csv_out = 'xgb-bayes-scores.csv')
    return best_param, best_score


def xgb_grid_search(stack_setting_,
                    exp, 
                    param_keys=None, param_vals=None,
                    k_fold=None):

    if param_keys is None:
        param_keys = ['model_type', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
                      'learning_rate', 'silent', 'objective', 'nthread', 'n_estimators', 'seed']
    if param_vals is None:
        param_vals = [[XGBClassifier], [4, 5, 6, 7, 8], [3, 4, 5, 6], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], [0.5, 0.6, 0.7, 0.8, 0.85, 0.9] ,
                      [0.01, 0.02, 0.03, 0.04], [1], ['binary:logistic'], [num_proc], [350, 450], [9438]]
    if k_fold is None:
        k_fold = 5

    gs = param_search.GridSearch(SklearnModel, exp, param_keys, param_vals,
                                 cv_folder = stack_setting_['1-Level']['xgb']['cv']['folder'],
                                 cv_out = stack_setting_['1-Level']['xgb']['cv']['cv_out'], 
                                 cv_pred_out = stack_setting_['1-Level']['xgb']['cv']['cv_pred_out'], 
                                 refit_pred_out = stack_setting_['1-Level']['xgb']['cv']['refit_pred_out'])
    best_param, best_score = gs.search_by_cv(validation_metrics = stack_setting_['1-Level']['xgb']['cv']['metrics'])

    # get meta_feature
    exp.write2csv_meta_feature(
        model = XGBClassifier(),
        meta_folder = stack_setting_['1-Level']['xgb']['meta_feature']['folder'],
        meta_train_fname = stack_setting_['1-Level']['xgb']['meta_feature']['train'],
        meta_test_fname = stack_setting_['1-Level']['xgb']['meta_feature']['test'],
        meta_header = stack_setting_['1-Level']['xgb']['meta_feature']['header'],
        best_param_ = best_param
        )

    # best_param is
    #  {'colsample_bytree': 0.7, 'silent': 1, 'learning_rate': 0.01, 'nthread': 6, 
    #  'min_child_weight': 6.0, 'n_estimators': 10, 'subsample': 0.5, 'seed': 9438, 
    #  'objective': 'binary:logistic', 'max_depth': 8}
    best_param['model_type'] = XGBClassifier 
    # commetout
    #param_search.write_cv_res_csv(cv_out = 'xgb-grid-scores2.pkl', 
    #                              cv_csv_out = 'xgb-grid-scores2.csv')

    # feature importance plot
    get_xgb_feature_importance_plot(best_param_ = best_param,
                                    experiment_ = exp,
                                    png_folder = stack_setting_['1-Level']['xgb']['graph']['folder'],
                                    png_fname = stack_setting_['1-Level']['xgb']['graph']['name'])

    return best_param, best_score


def get_xgb_feature_importance_plot(best_param_, experiment_, 
                                    png_folder,
                                    png_fname,
                                    score_threshold=0.8):

    # 1. 
    train_X, train_y = experiment_.get_train_data()
    clf = XGBClassifier()
    try:
        del best_param_['model_type']
    except:
        pass
    clf.set_params(**best_param_)
    clf.fit(train_X, train_y)
    index2feature = clf.booster().get_fscore()
    fis = pd.DataFrame({'name':index2feature.keys(),
                        'score':index2feature.values()})
    fis = fis.sort('score', ascending=False)
    if len(fis.index) > 20:
        score_threshold = fis['score'][fis['score'] > 0.0].quantile(score_threshold)
        #where_str = 'score > %f & score > %f' % (score_threshold, 0.0)
        where_str = 'score >= %f' % (score_threshold)
        fis = fis.query(where_str)

    # 2. plot
    #gs = GridSpec(2,2)
    #ax1 = plt.subplot(gs[:,0])
    #ax2 = plt.subplot(gs[0,1])
    #ax3 = plt.subplot(gs[1,1])

    # 3.1 feature importance
    sns.barplot(x = 'score', y = 'name',
                data = fis,
                #ax=ax1,
                color="blue")
    #plt.title("Feature_Importance", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.xlabel("Feature_Importance : f-Score", fontsize=10)

    """
    # 3.2 PDF
    confidence_score = clf.oob_decision_function_[:,1]
    sns.distplot(confidence_score, kde=False, rug=False, ax=ax2)
    ax2.set_title("PDF")

    # 3.3 CDF
    num_bins = min(best_param_.get('n_estimators',1), 100)
    counts, bin_edges = np.histogram(confidence_score, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    ax3.plot(bin_edges[1:], cdf / cdf.max())
    ax3.set_title("CDF")
    ax3.set_xlabel("Oob_Decision_Function:Confidence_Score", fontsize=10)
    """

    png_fname = os.path.join(Config.get_string('data.path'), 'graph', png_fname)
    plt.tight_layout()
    plt.savefig(png_fname)#, bbox_inches='tight', pad_inches=1)
    plt.close()

    return True


def xgb_submmision(exp, param=None):
    if not param:
        param = {'colsample_bytree': 0.6923529515220681, 'silent': 1, 'model_type':XGBClassifier, 'learning_rate': 0.014582411837608816, 'nthread': 4, 'min_child_weight': 6.0, 'n_estimators': 400, 'subsample': 0.5530324529773664, 'seed': 9438, 'objective': 'binary:logistic', 'max_depth': 8.0}
    xgb_model = SklearnModel(param)
    final_preds = exp.fit_fullset_and_predict(xgb_model)
    submission_path = os.path.join(Config.get_string('data.path'), 'submission')
    fname = os.path.join(submission_path, xgb_model.to_string().split("-")[0] + '_res.csv')
    #fname = os.path.join(submission_path, 'xgb_bayes_param_res.csv')
    #print final_preds
    #print exp.test_id
    save_submissions(fname, exp.test_id, final_preds)
