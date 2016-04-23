# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
import sys, os
from sklearn.ensemble import RandomForestClassifier
#from sklearn.pipeline import Pipeline
#from sklearn.grid_search import GridSearchCV
from experiment.stacking.experiment_l1 import ExperimentL1
from experiment.param_search import GridSearch
from experiment.model_wrappers import SklearnModel
from utils.config_utils import Config

import sklearn.cross_validation as cross_validation
from sklearn.base import BaseEstimator, TransformerMixin, clone

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pandas as pd
import numpy as np

def rf_grid_search(stack_setting_,
                   param_keys=None, param_vals=None,
                   num_proc=None):

    if stack_setting_ is None:
        sys.stderr.write('You have no setting Json file\n')
        sys.exit()

    if num_proc is None:
        num_proc = 6

    if param_keys is None:
        param_keys = ['model_type', 'n_estimators', 'criterion', 'n_jobs']

    if param_vals is None:
        param_vals = [[RandomForestClassifier], [500], ['gini', 'entropy'],  [num_proc]]

    #exp = ExperimentL1()
    exp = ExperimentL1(data_folder = stack_setting_['0-Level']['folder'],
                       train_fname = stack_setting_['0-Level']['train'], 
                       test_fname = stack_setting_['0-Level']['test'])
    gs = GridSearch(SklearnModel, exp, param_keys, param_vals,
                    cv_folder = stack_setting_['1-Level']['rf']['cv']['folder'],
                    cv_out = stack_setting_['1-Level']['rf']['cv']['cv_out'], 
                    cv_pred_out = stack_setting_['1-Level']['rf']['cv']['cv_pred_out'], 
                    refit_pred_out = stack_setting_['1-Level']['rf']['cv']['refit_pred_out'])
    best_param, best_score = gs.search_by_cv(validation_metrics = stack_setting_['1-Level']['rf']['cv']['metrics'])

    # get meta_feature
    exp.write2csv_meta_feature(
        model = RandomForestClassifier(),
        meta_folder = stack_setting_['1-Level']['rf']['meta_feature']['folder'],
        meta_train_fname = stack_setting_['1-Level']['rf']['meta_feature']['train'],
        meta_test_fname = stack_setting_['1-Level']['rf']['meta_feature']['test'],
        meta_header = stack_setting_['1-Level']['rf']['meta_feature']['header'],
        best_param_ = best_param
        )

    # get feature importance plot
    get_rf_feature_importance_plot(best_param_ = best_param, 
                                   experiment_ = exp, 
                                   png_folder = stack_setting_['1-Level']['rf']['graph']['folder'],
                                   png_fname = stack_setting_['1-Level']['rf']['graph']['name'])

    return best_param, best_score


def get_rf_feature_importance_plot(best_param_, experiment_, 
                                   png_folder,
                                   png_fname,
                                   score_threshold=0.8):

    # 1. 
    best_param_['oob_score'] = True

    # 2. 
    train_X, train_y = experiment_.get_train_data()
    clf = RandomForestClassifier()
    clf.set_params(**best_param_)
    clf.fit(train_X, train_y)

    index2feature = dict(zip(np.arange(len(train_X.columns.values)), train_X.columns.values))
    feature_importances_index = [str(j) for j in clf.feature_importances_.argsort()[::-1]]
    feature_importances_score = [clf.feature_importances_[int(j)] for j in feature_importances_index]
    fis = pd.DataFrame(
        {'name':[index2feature.get(int(key),'Null') for key in feature_importances_index],
         'score':feature_importances_score}
        )
    if len(fis.index) > 20:
        score_threshold = fis['score'][fis['score'] > 0.0].quantile(score_threshold)
        #where_str = 'score > %f & score > %f' % (score_threshold, 0.0)
        where_str = 'score >= %f' % (score_threshold)
        fis = fis.query(where_str)
    
    # 3. plot
    gs = GridSpec(2,2)
    ax1 = plt.subplot(gs[:,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,1])

    # 3.1 feature importance
    sns.barplot(x = 'score', y = 'name',
                data = fis,
                ax=ax1,
                color="blue")
    #ax1.set_title("Feature_Importance", fontsize=10)
    ax1.set_ylabel("Feature", fontsize=10)
    ax1.set_xlabel("Feature_Importance", fontsize=10)

    # 3.2 PDF
    confidence_score = clf.oob_decision_function_[:,1]
    sns.distplot(confidence_score, kde=False, rug=False, ax=ax2)
    ax2.set_title("PDF")

    # 3.3 CDF
    num_bins = 100
    try:
        counts, bin_edges = np.histogram(confidence_score, bins=num_bins, normed=True)
    except:
        counts, bin_edges = np.histogram(confidence_score, normed=True)
    cdf = np.cumsum(counts)
    ax3.plot(bin_edges[1:], cdf / cdf.max())
    ax3.set_title("CDF")
    ax3.set_xlabel("Oob_Decision_Function:Confidence_Score", fontsize=10)

    png_fname = os.path.join(Config.get_string('data.path'), png_folder, png_fname)
    plt.tight_layout()
    plt.savefig(png_fname)
    plt.close()

    return True

"""
def selectKImportanceFeature(model, X, k=5):
    return X[:, model.feature_importances_.argsort()[::-1][:k]]


class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
        self.model = model
        self.n = n

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X):
        return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]
"""

