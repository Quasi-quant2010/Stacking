# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics

from utils.config_utils import Config

class ExperimentL2:
    """
    Level 2 experiment wrapper for model blending
    """

    def __init__(self, 
                 data_folder,
                 train_fname=None, test_fname=None):
        self.random_state = 325243  # do not change it for different l2 models!
        #self.random_state = 98754  # do not change it for different l2 models!
        if not train_fname:
            sys.stderr.write('Do not set train_meta_feature\n')
            sys.exit()
        if not test_fname:
            sys.stderr.write('Do not set test_meta_feature\n')
            sys.exit()
        train_fname = os.path.join(Config.get_string('data.path'), data_folder, train_fname)
        test_fname = os.path.join(Config.get_string('data.path'), data_folder, test_fname)

        # load train data
        train = pd.read_csv(train_fname)
        self.train_id = train.values
        self.train_y = train.label.values
        self.train_x = train.drop(['label'], axis=1)
        # load test data
        test = pd.read_csv(test_fname)
        self.test_id = test.values
        self.test_y = test.label.values
        self.test_x = test.drop(['label'], axis=1)
        #print self.train_x.head()
        #print self.test_x.head()

    def cross_validation(self, model):
        # kfold = cross_validation.KFold(self.train_x.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=5, shuffle=True, random_state=self.random_state)
        scores = {'auc':list(),
                  'hinge_loss':list(),
                  'log_loss':list(),
                  'accuracy':list(),
                  'precision':list(),
                  'recall':list(),
                  'f1_value':list()}
        #scores = list()
        preds = np.zeros(len(self.train_y))
        i = 0
        for train_idx, test_idx in kfold:
            print (' --------- fold {0} ---------- '.format(i))
            train_x = self.train_x.iloc[train_idx] # 明示的にindex, columsを番号で指定したい, sinhrks.hatenablog.com/entry/2014/11/12/233216
            train_y = self.train_y[train_idx]
            test_x = self.train_x.iloc[test_idx]
            test_y = self.train_y[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred)
            preds[test_idx] = pred

            score = metrics.roc_auc_score(test_y, pred)# auc
            scores['auc'].append(score)
            score = metrics.hinge_loss(test_y, pred)# hinge_loss
            scores['hinge_loss'].append(score)
            score = metrics.log_loss(test_y, pred)# log_loss
            scores['log_loss'].append(score)
            #score = metrics.accuracy_score(test_y, pred)# accuracy
            #scores['accuracy'].append(score)
            #score = metrics.precision_score(test_y, pred)# precision
            #scores['precision'].append(score)
            #score = metrics.recall_score(test_y, pred)# recall
            #scores['recall'].append(score)
            #score = metrics.f1_score(test_y, pred)# f_value
            #scores['f1_value'].append(score)
            i += 1
        for key in scores.keys():
            scores[key] = np.asarray(scores[key], dtype=np.float32)

        #print scores.mean(), scores.std()
        return scores, preds

    def fit_fullset_and_predict(self, model):
        model.fit(self.train_x, self.train_y)
        preds = model.predict(self.test_x)
        return preds

    def get_proba(self,
                  clf, X):
        # [reference]
        # Probability Calinration Curves in sklearn
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X)[:,1]
        else: # use decision function
            prob_pos = clf.decision_function(X)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min()) # scailing [0,1]

        return prob_pos

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

    def get_data_col_name(self):
        return self.train_x.columns.values
