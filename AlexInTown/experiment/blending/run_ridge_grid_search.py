# -*- coding:utf-8 -*-

import os, gzip, cPickle
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error
from experiment.blending.experiment_blend import ExperimentL2
from experiment.param_search import GridSearch
from experiment.model_wrappers import SklearnModel
from experiment.metrics import precision_recall
from utils.config_utils import Config

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pandas as pd
import numpy as np

# predict evaluation
def ridge_blend(stack_setting_, best_param_):

    folder = stack_setting_['2-Level']['blending']['folder']
    blend_weight_fname = stack_setting_['2-Level']['blending']['weight']
    blend_weight_fname = os.path.join(Config.get_string('data.path'), folder, blend_weight_fname)
    linear_weight = pd.read_csv(blend_weight_fname)

    folder = stack_setting_['1-Level']['meta_features']['folder']
    test_fname = stack_setting_['1-Level']['meta_features']['test']
    test_fname = os.path.join(Config.get_string('data.path'), folder, test_fname)
    test = pd.read_csv(test_fname)

    folder = stack_setting_['2-Level']['blending']['folder']
    model_fname = stack_setting_['2-Level']['blending']['model']
    model_fname = os.path.join(Config.get_string('data.path'), folder, model_fname)
    with gzip.open(model_fname, 'rb') as gf:
        model = cPickle.load(gf)

    y_test = test.label.values
    X_test = test.drop(['label'], axis = 1)
    del test

    y_predict = model.predict(X_test)

    #return mean_squared_error(y_test, y_predict)
    return precision_recall(y_test, y_predict)


# blending
def ridge_grid_search(stack_setting_,
                      param_keys=None, param_vals=None,
                      num_proc=None):

    if stack_setting_ is None:
        sys.stderr.write('You have no setting Json file\n')
        sys.exit()

    if num_proc is None:
        num_proc = 6

    if param_keys is None:
        param_keys = ['model_type', 'alpha', 'solver', 'max_iter', 'random_state', 'fit_intercept']

    if param_vals is None:
        param_vals = [[RidgeClassifier], [1.0], ['auto', 'sag'],
                      [100], [30], [True]]


    #exp = ExperimentL2(data_folder = stack_setting_['1-Level']['meta_features']['folder'],
    #                   train_fname = stack_setting_['1-Level']['meta_features']['train'], 
    #                   test_fname = stack_setting_['1-Level']['meta_features']['test'])
    exp = ExperimentL2(data_folder = stack_setting_['2-Level']['blending']['folder'],
                       train_fname = stack_setting_['2-Level']['blending']['ptrain'], 
                       test_fname = stack_setting_['2-Level']['blending']['ptest'])

    gs = GridSearch(SklearnModel, exp, param_keys, param_vals,
                    cv_folder = stack_setting_['2-Level']['ridge']['cv']['folder'],
                    cv_out = stack_setting_['2-Level']['ridge']['cv']['cv_out'], 
                    cv_pred_out = stack_setting_['2-Level']['ridge']['cv']['cv_pred_out'], 
                    refit_pred_out = stack_setting_['2-Level']['ridge']['cv']['refit_pred_out'])
    best_param, best_score = gs.search_by_cv(validation_metrics = stack_setting_['2-Level']['ridge']['cv']['metrics'])

    # get best linear blending weight
    get_optimal_blend_weigth(exp_ = exp, best_param_ = best_param,
                             folder = stack_setting_['2-Level']['blending']['folder'],
                             fname = stack_setting_['2-Level']['blending']['weight'],
                             model_fname = stack_setting_['2-Level']['blending']['model'])

    # get regularization path
    # ridge coefficients as a function of the refulatization
    get_ridge_plot(best_param_ = best_param, experiment_ = exp, 
                   param_keys_ = param_keys, param_vals_ = param_vals,
                   png_folder = stack_setting_['2-Level']['ridge']['graph']['folder'],
                   png_fname = stack_setting_['2-Level']['ridge']['graph']['name'])

    return best_param, best_score


def get_optimal_blend_weigth(exp_, best_param_,
                             folder, fname, model_fname):
    clf = RidgeClassifier()
    X_test, y_test = exp_.get_test_data()
    clf.set_params(**best_param_)
    clf.fit(X_test, y_test)

    # dump2csv optimal linear weight
    names = np.append(np.array(['intercept'], dtype='S100'), X_test.columns.values)
    coefs = np.append(clf.intercept_, clf.coef_).astype(np.float64)
    optimal_linear_weight = pd.DataFrame(coefs.reshape(1,len(coefs)), columns=names)
    optimal_linear_weight.to_csv(os.path.join(Config.get_string('data.path'),
                                              folder,
                                              fname), index=False)

    # dump2cpkle for ridge model
    model_fname = os.path.join(Config.get_string('data.path'), folder, model_fname)
    with gzip.open(model_fname, 'wb') as gf:
        cPickle.dump(clf, gf, cPickle.HIGHEST_PROTOCOL)
    
    return True


def get_ridge_plot(best_param_, experiment_, 
                   param_keys_, param_vals_,
                   png_folder,
                   png_fname,
                   score_threshold=0.8):

    parameters = dict(zip(param_keys_, param_vals_))
    del parameters['model_type']

    clf = RidgeClassifier()
    X_train, y_train = experiment_.get_train_data()
    clf.set_params(**best_param_)
    clf.fit(X_train, y_train)    
    best_alpha = best_param_['alpha']
    result = {'alphas':[],
              'coefs':np.zeros( (len(parameters['alpha']), len(X_train.columns.values) + 1) ),
              'scores':[],
              'score':None}


    for i, alpha in enumerate(parameters.get('alpha',None)):
        result['alphas'].append(alpha)
        del best_param_['alpha']
        best_param_['alpha'] = alpha
        clf.set_params(**best_param_)
        clf.fit(X_train, y_train)

        # regularization path
        tmp = np.array([0 for j in xrange(len(X_train.columns.values) + 1)], dtype=np.float32)
        if best_param_['fit_intercept']:
            tmp = np.append(clf.intercept_, clf.coef_)
        else:
            tmp[1:] = clf.intercept_
        result['coefs'][i,:] = tmp
        result['scores'].append(experiment_.get_proba(clf, X_train))
    del X_train, y_train

    # 2. 
    tmp_len = len(experiment_.get_data_col_name())
    index2feature = dict(zip(np.arange(1, tmp_len + 1), 
                             experiment_.get_data_col_name()))
    if best_param_['fit_intercept']:
        index2feature[0] = 'intercept'

    # 3. plot
    gs = GridSpec(2,2)
    ax1 = plt.subplot(gs[:,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,1])


    # 3.1 feature importance
    labels = np.append(np.array(['intercept'], dtype='S100'), experiment_.get_data_col_name())
    nrows, ncols = result['coefs'].shape
    for ncol in xrange(ncols):
        ax1.plot(np.array(result['alphas']), result['coefs'][:,ncol], label = labels[ncol])
    ax1.legend(loc='best')
    ax1.set_xscale('log')
    ax1.set_title("Regularization Path:%1.3e" % (best_alpha))
    ax1.set_xlabel("alpha", fontsize=10)

    # 3.2 PDF
    X_test, y_test = experiment_.get_test_data()
    result['score'] = clf.decision_function(X_test)
    sns.distplot(result['score'], kde=False, rug=False, ax=ax2)
    ax2.set_title("PDF : Decision_Function")


    # 3.3 CDF
    num_bins = 100
    try:
        counts, bin_edges = np.histogram(result['score'], bins=num_bins, normed=True)
    except:
        counts, bin_edges = np.histogram(result['score'], normed=True)

    cdf = np.cumsum(counts)
    ax3.plot(bin_edges[1:], cdf / cdf.max())
    ax3.set_title("CDF")
    ax3.set_xlabel("Decision_Function:Confidence_Score", fontsize=10)


    png_fname = os.path.join(Config.get_string('data.path'), png_folder, png_fname)
    plt.tight_layout()
    plt.savefig(png_fname)
    plt.close()

    return True
