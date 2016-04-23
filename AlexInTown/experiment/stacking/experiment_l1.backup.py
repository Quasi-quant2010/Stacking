# -*- coding: utf-8 -*-
import os
import gzip,cPickle
import numpy as np
import pandas as pd
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics

from experiment.metrics import precision_recall
from utils.config_utils import Config


class ExperimentL1:
    """
    Level 1 experiment wrapper for model stacking.
    """

    def __init__(self, 
                 data_folder,
                 train_fname=None, test_fname=None):
        #self.random_state = 325243  # do not change it for different l1 models!
        self.random_state = 98754  # do not change it for different l1 models!
        if not train_fname:
            train_fname = 'filtered_train.csv'
        if not test_fname:
            test_fname = 'filtered_test.csv'
        train_fname = os.path.join(Config.get_string('data.path'), data_folder, train_fname)
        test_fname = os.path.join(Config.get_string('data.path'), data_folder, test_fname)
        # load train data
        train = pd.read_csv(train_fname)
        train.sort(columns='ID', inplace=1)
        self.train_id = train.values
        self.train_y = train.TARGET.values
        self.train_x = train.drop(['ID', 'TARGET'], axis=1)
        # load test data
        test = pd.read_csv(test_fname)
        test.sort(columns='ID', inplace=1)
        self.test_id = test.values
        self.test_y = test.TARGET.values
        #self.test_x = test.drop(['ID'], axis=1)
        self.test_x = test.drop(['ID', 'TARGET'], axis=1)
        #print self.train_x.head()
        #print self.test_x.head()


    def cross_validation(self, model):
        # kfold = cross_validation.KFold(self.train_x.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=5, shuffle=True, random_state=self.random_state)
        #scores = {'auc':list(),
        #          'hinge_loss':list(),
        #          'log_loss':list(),
        #          'accuracy':list(),
        #          'precision':list(),
        #          'recall':list(),
        #          'f_value':list()}
        scores = list()
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
            #print precision_recall(test_y, pred)
            preds[test_idx] = pred
            scores.append(score)
            i += 1
        scores = np.asarray(scores)
        #print scores.mean(), scores.std()
        return scores, preds

    def fit_fullset_and_predict(self, model):
        model.fit(self.train_x, self.train_y)
        preds = model.predict(self.test_x)
        return preds

    def write2csv_meta_feature(self,
                               model,
                               meta_folder, meta_train_fname, meta_test_fname, meta_header,
                               best_param_):

        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=5, shuffle=True, random_state=self.random_state)

        model.set_params(**best_param_)

        transform_train = np.zeros((self.train_x.shape[0],2), dtype=np.float32)
        
        transform_test = np.zeros((self.test_x.shape[0], 2), dtype=np.float32)

        # transform train data
        for i, (train_idx, test_idx) in enumerate(kfold):
            print (' [Meta Feature] --------- fold {0} ---------- '.format(i))
            train_x = self.train_x.iloc[train_idx]
            train_y = self.train_y[train_idx]
            test_x = self.train_x.iloc[test_idx]
            test_y = self.train_y[test_idx]
            model.fit(train_x, train_y)
            #transform_train[test_idx, 0] = model.predict_proba(test_x)[:,1].astype(np.float32)w
            transform_train[test_idx, 0] = self.get_proba(model, test_x).astype(np.float32)
            transform_train[test_idx, 1] = test_y.astype(np.int32)

        meta_train_fname = os.path.join(Config.get_string('data.path'), 
                                        meta_folder, 
                                        meta_train_fname)
        np.savetxt(meta_train_fname, transform_train, delimiter=',',
                   header=meta_header, comments='', fmt='%1.10e,%d')
        del transform_train

        # transform test data
        model.fit(self.train_x, self.train_y)
        #transform_test = model.predict_proba(self.test_x)[:,1].astype(np.float32)
        transform_test[:,0] = self.get_proba(model, self.test_x).astype(np.float32)
        transform_test[:,1] = self.test_y.astype(np.int32)
        meta_test_fname = os.path.join(Config.get_string('data.path'), 
                                       meta_folder, 
                                       meta_test_fname)
        np.savetxt(meta_test_fname, transform_test, delimiter=',',
                   header=meta_header, comments='', fmt='%1.10e,%d')
        del transform_test

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


    def param_selection(self, params):
        pass


class ExperimentL1_1:
    """
    Level 1_1 experiment wrapper for model stacking.
    Ensemble + Linear Classifier
    """

    def __init__(self, 
                 data_folder,
                 train_fname=None, test_fname=None):
        #self.random_state = 325243  # do not change it for different l1 models!
        self.random_state = 98754  # do not change it for different l1 models!
        if not train_fname:
            train_fname = 'filtered_train.csv'
        if not test_fname:
            test_fname = 'filtered_test.csv'
        train_fname = os.path.join(Config.get_string('data.path'), data_folder, train_fname)
        test_fname = os.path.join(Config.get_string('data.path'), data_folder, test_fname)

        # load train data
        with gzip.open(train_fname, 'rb') as gf:
            self.train_x, self.train_y = cPickle.load(gf)

        # load test data
        with gzip.open(test_fname, 'rb') as gf:
            self.test_x, self.test_y = cPickle.load(gf)


    def cross_validation(self, model):
        # kfold = cross_validation.KFold(self.train_x.shape[0], n_folds=5, shuffle=True, random_state=self.random_state)
        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=5, shuffle=True, random_state=self.random_state)
        scores = list()
        preds = np.zeros(len(self.train_y))
        i = 0
        for train_idx, test_idx in kfold:
            print (' --------- fold {0} ---------- '.format(i))
            train_x = self.train_x.toarray()[train_idx]
            train_y = self.train_y[train_idx]
            test_x = self.train_x.toarray()[test_idx]
            test_y = self.train_y[test_idx]
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
            score = metrics.roc_auc_score(test_y, pred)
            preds[test_idx] = pred
            scores.append(score)
            i += 1
        scores = np.asarray(scores)
        #print scores.mean(), scores.std()
        return scores, preds

    def fit_fullset_and_predict(self, model):
        model.fit(self.train_x, self.train_y)
        preds = model.predict(self.test_x)
        return preds

    def write2csv_meta_feature(self,
                               model,
                               meta_folder, meta_train_fname, meta_test_fname, meta_header,
                               best_param_):

        kfold = cross_validation.StratifiedKFold(self.train_y, n_folds=5, shuffle=True, random_state=self.random_state)

        model.set_params(**best_param_)
        transform_train = np.zeros((self.train_x.shape[0],2), dtype=np.float32)        
        transform_test = np.zeros((self.test_x.shape[0], 2), dtype=np.float32)

        # transform train data
        for i, (train_idx, test_idx) in enumerate(kfold):
            print (' [Meta Feature] --------- fold {0} ---------- '.format(i))
            train_x = self.train_x.toarray()[train_idx]
            train_y = self.train_y[train_idx]
            test_x = self.train_x.toarray()[test_idx]
            test_y = self.train_y[test_idx]
            model.fit(train_x, train_y)
            #transform_train[test_idx, 0] = model.predict_proba(test_x)[:,1].astype(np.float32)
            transform_train[test_idx, 0] = self.get_proba(model, test_x).astype(np.float32)
            transform_train[test_idx, 1] = test_y.astype(np.int32)

        meta_train_fname = os.path.join(Config.get_string('data.path'), 
                                        meta_folder, 
                                        meta_train_fname)
        np.savetxt(meta_train_fname, transform_train, delimiter=',',
                   header=meta_header, comments='', fmt='%1.10e,%d')
        del transform_train

        # transform test data
        model.fit(self.train_x, self.train_y)
        #transform_test = model.predict_proba(self.test_x)[:,1].astype(np.float32)
        transform_test[:,0] = self.get_proba(model, self.test_x).astype(np.float32)
        transform_test[:,1] = self.test_y.astype(np.int32)
        meta_test_fname = os.path.join(Config.get_string('data.path'), 
                                       meta_folder, 
                                       meta_test_fname)
        np.savetxt(meta_test_fname, transform_test, delimiter=',',
                   header=meta_header, comments='', fmt='%1.10e,%d')
        del transform_test

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


    def param_selection(self, params):
        pass

