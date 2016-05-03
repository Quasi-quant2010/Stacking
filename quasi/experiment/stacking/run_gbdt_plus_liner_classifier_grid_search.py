# -*- coding:utf-8 -*-

import os, sys, gzip, cPickle
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV

from experiment.stacking.experiment_l1 import ExperimentL1, ExperimentL1_1
from experiment.param_search import GridSearch
from experiment.model_wrappers import SklearnModel, TreeTransform
from utils.config_utils import Config


def gbdt_plus_liner_classifier_grid_search(stack_setting_,
                                           upper_param_keys=None, upper_param_vals=None,
                                           lower_param_keys=None, lower_param_vals=None,
                                           k_fold=None):

    """
     upper model is GBDT or Random Forest
     lower model is Linear Classifier
    """
    if stack_setting_ is None:
        sys.stderr.write('You have no setting Json file\n')
        sys.exit()

    if k_fold is None:
        k_fold = 5

    upper_best_params = None
    lower_best_param = None


    # 1. upper model
    if upper_param_keys is None:
        upper_param_keys = ['model_type', 'n_estimators', 'loss', 'random_state', 'subsample', 'max_features', 'max_leaf_nodes', 'learning_rate', 'max_depth', 'min_samples_leaf']

    if upper_param_vals is None:
        upper_param_vals = [[GradientBoostingClassifier], [100], ['deviance'], [0], [0.1], [5], [20], [0.1], [2], [8]]


    # grid search for upper model : GBDT or Random Forest
    # ExperimentL1 has model free. On the other hand, data is fix
    exp = ExperimentL1(data_folder = stack_setting_['0-Level']['folder'],
                       train_fname = stack_setting_['0-Level']['train'], 
                       test_fname = stack_setting_['0-Level']['test'],
                       k_fold_ = k_fold)


    model_folder = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['folder']
    model_train_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['train']
    model_train_fname = os.path.join(Config.get_string('data.path'), 
                                     model_folder, 
                                     model_train_fname)
    model_folder = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['folder']
    model_test_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['test']
    model_test_fname = os.path.join(Config.get_string('data.path'), 
                                    model_folder, 
                                    model_test_fname)
    upper_param_dict = dict(zip(upper_param_keys, upper_param_vals))
    if os.path.isfile(model_train_fname) is False and \
            os.path.isfile(model_test_fname) is False:
        #upper_param_dict['model_type'] == [GradientBoostingClassifier]
        del upper_param_dict['model_type']
        # data
        X_train, y_train = exp.get_train_data()
        X_test, y_test = exp.get_test_data()
        
        # Upper Model
        clf = TreeTransform(estimator = GradientBoostingClassifier(),
                            X_train_ = X_train, y_train_ = y_train, 
                            X_test_ = X_test, y_test_ = y_test,
                            param_candidate = upper_param_dict)
        upper_best_params = clf.get_best_param()

        # Fit
        if type(X_train) == pd.core.frame.DataFrame:
            clf.fit(X_train.as_matrix().astype(np.float32), y_train)
        elif X_train == np.ndarray:
            clf.fit(X_train.astype(np.float32), y_train)

        # train result
        train_loss = clf.estimator_.train_score_
        test_loss = np.zeros((len(clf.estimator_.train_score_),), dtype=np.float64)
        for iter_, y_pred in enumerate(clf.estimator_.staged_decision_function(X_test)):
            test_loss[iter_] = clf.estimator_.loss_(y_test, y_pred)
            #print iter_, clf.estimator_.train_score_[iter_], test_loss[iter_]

        gs = GridSpec(2,2)
        #ax1 = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[:,1])
        ax3 = plt.subplot(gs[:,0])

        # generalization error
        ax2.plot(train_loss, label="train_loss")
        ax2.plot(test_loss, label="test_loss")
        ax2.set_xlabel('Boosting Iterations')
        ax2.set_ylabel('%s Loss' % (upper_best_params.get('loss','RMSE')))
        ax2.legend(loc="best")

        # tree ensembles
        score_threshold=0.8
        index2feature = dict(zip(np.arange(len(X_train.columns.values)), X_train.columns.values))
        feature_importances_index = [str(j) for j in clf.estimator_.feature_importances_.argsort()[::-1]]
        feature_importances_score = [clf.estimator_.feature_importances_[int(j)] for j in feature_importances_index]
        fis = pd.DataFrame(
            {'name':[index2feature.get(int(key),'Null') for key in feature_importances_index],
             'score':feature_importances_score}
            )
        if len(fis.index) > 20:
            score_threshold = fis['score'][fis['score'] > 0.0].quantile(score_threshold)
            # where_str = 'score > %f & score > %f' % (score_threshold, 0.0)
            where_str = 'score >= %f' % (score_threshold)
            fis = fis.query(where_str)

        sns.barplot(x = 'score', y = 'name',
                    data = fis,
                    ax=ax3,
                    color="blue")
        ax3.set_xlabel("Feature_Importance", fontsize=10)
        plt.tight_layout()

        graph_folder = stack_setting_['1-Level']['gbdt_linear']['upper']['graph']['folder']
        graph_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['graph']['name']
        graph_fname = os.path.join(Config.get_string('data.path'), 
                                   graph_folder, 
                                   graph_fname)
        plt.savefig(graph_fname)
        plt.close()


        # feature transform
        #print clf.toarray().shape
        # >(26049, 100)
        #input_features = 26049, weak_learners = 100
        #print len(one_hot.toarray()[:,0]), one_hot.toarray()[:,0]
        #print len(one_hot.toarray()[0,:]), one_hot.toarray()[0,:]

        ## feature transformation : get test data from train trees
        #print transformated_train_features.shape, X_train.shape
        #print transformated_test_features.shape, X_test.shape

        transformated_train_features = clf.one_hot_encoding
        if type(X_test) == pd.core.frame.DataFrame:
            transformated_test_features = clf.transform(X_test.as_matrix().astype(np.float32), 
                                                        y_test)
        elif type(X_train) == np.ndarray:
            transformated_test_features = clf.transform(X_test, y_test)

        #model_folder = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['folder']
        #model_train_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['train']
        #model_train_fname = os.path.join(Config.get_string('data.path'), 
        #                                 model_folder, 
        #                                 model_train_fname)
        with gzip.open(model_train_fname, "wb") as gf:
            cPickle.dump([transformated_train_features, y_train], 
                         gf,
                         cPickle.HIGHEST_PROTOCOL)

        #model_folder = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['folder']
        #model_test_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['test']
        #model_test_fname = os.path.join(Config.get_string('data.path'), 
        #                                model_folder, 
        #                                model_test_fname)
        with gzip.open(model_test_fname, "wb") as gf:
            cPickle.dump([transformated_test_features, y_test],
                         gf,
                         cPickle.HIGHEST_PROTOCOL)


    # 2. lower model
    if lower_param_keys is None:
        lower_param_keys = ['model_type', 'n_neighbors', 'weights',
                            'algorithm', 'leaf_size', 'metric', 'p', 'n_jobs']

    if lower_param_vals is None:
        lower_param_vals = [[KNeighborsClassifier], [1, 2, 4, 8, 16, 24, 32, 64], ['uniform', 'distance'],
                            ['ball_tree'], [30], ['minkowski'], [2], [4]]

    lower_param_dict = dict(zip(lower_param_keys, lower_param_vals))
    clf_lower_model = None
    clf_lower_mname = None

    # grid search for lower model : Linear Classifier
    # ExperimentL1_1 has model free. On the other hand, data is fix
    if lower_param_dict['model_type'] == [LogisticRegression] and lower_param_dict['penalty'] == ['l1']:
        # Logistic Regression
        clf_lower_model = LogisticRegression()
        clf_lower_mname = 'LR-L1'

    elif lower_param_dict['model_type'] == [LogisticRegression] and lower_param_dict['penalty'] == ['l2']:
        # Logistic Regression
        clf_lower_model = LogisticRegression()
        clf_lower_mname = 'LR-L2'

    elif lower_param_dict['model_type'] == [LinearSVC] and lower_param_dict['penalty'] == ['l1']:
        # SVM L1
        clf_lower_model = LinearSVC()
        clf_lower_mname = 'SVM-L1'

    elif lower_param_dict['model_type'] == [LinearSVC] and lower_param_dict['penalty'] == ['l2']:
        # SVM L1
        clf_lower_model = LinearSVC()
        clf_lower_mname = 'SVM-L2'

    else:
        sys.stderr.write("You should input lower liner model\n")
        sys.exit()

    model_train_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['train']
    model_test_fname = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['test']
    exp = ExperimentL1_1(data_folder = stack_setting_['1-Level']['gbdt_linear']['upper']['gbdt']['folder'],
                         train_fname = model_train_fname, 
                         test_fname = model_test_fname,
                         k_fold_ = k_fold)
    # GridSearch has a single model. model is dertermined by param
    gs = GridSearch(SklearnModel, exp, lower_param_keys, lower_param_vals,
                    cv_folder = stack_setting_['1-Level']['gbdt_linear']['lower']['cv']['folder'],
                    cv_out = stack_setting_['1-Level']['gbdt_linear']['lower']['cv']['cv_out'], 
                    cv_pred_out = stack_setting_['1-Level']['gbdt_linear']['lower']['cv']['cv_pred_out'], 
                    refit_pred_out = stack_setting_['1-Level']['gbdt_linear']['lower']['cv']['refit_pred_out'])
    lower_best_param, lower_best_score = gs.search_by_cv(validation_metrics = stack_setting_['1-Level']['gbdt_linear']['lower']['cv']['metrics'])
    print lower_best_param

    # get meta_feature
    meta_train_fname_ = "%s_%s.%s" % (
        ".".join(stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['train'].split(".")[:-1]),
        clf_lower_mname,
        stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['train'].split(".")[-1]
        )
    meta_test_fname_ = "%s_%s.%s" % (
        ".".join(stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['test'].split(".")[:-1]),
        clf_lower_mname,
        stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['test'].split(".")[-1]
        )
    meta_header_ = "%s_%s,%s" % (
        ",".join(stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['header'].split(",")[:-1]),
        clf_lower_mname,
        stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['header'].split(",")[-1]
        )
    exp.write2csv_meta_feature(
        model = clf_lower_model,
        meta_folder = stack_setting_['1-Level']['gbdt_linear']['lower']['meta_feature']['folder'],
        meta_train_fname = meta_train_fname_,
        meta_test_fname = meta_test_fname_,
        meta_header = meta_header_,
        best_param_ = lower_best_param
        )

    ## best parameter for GBDT and anohter sklearn classifier
    #return best_param, best_score
    
    if upper_best_params is None:
        upper_best_params = stack_setting_['1-Level']['gbdt_linear']['upper']['best_parameter']

    return upper_best_params, lower_best_param
