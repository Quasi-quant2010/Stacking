# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.grid_search import GridSearchCV

from scipy.sparse import hstack


class TreeTransform(BaseEstimator, TransformerMixin):
    """One-hot encode samples with an ensemble of trees
    
    This transformer first fits an ensemble of trees (e.g. gradient
    boosted trees or a random forest) on the training set.

    Then each leaf of each tree in the ensembles is assigned a fixed
    arbitrary feature index in a new feature space. If you have 100
    trees in the ensemble and 2**3 leafs per tree, the new feature
    space has 100 * 2**3 == 800 dimensions.
    
    Each sample of the training set go through the decisions of each tree
    of the ensemble and ends up in one leaf per tree. The sample if encoded
    by setting features with those leafs to 1 and letting the other feature
    values to 0.
    
    The resulting transformer learn a supervised, sparse, high-dimensional
    categorical embedding of the data.
    
    This transformer is typically meant to be pipelined with a linear model
    such as logistic regression, linear support vector machines or
    elastic net regression.

    [Reference]
    Agaclar Ile Oge Kodlamasi(Categorical Embedding)
    http://sayilarvekuramlar.blogspot.jp/2014/11/agaclar-ile-oge-kodlamasi-categorical.html
    """

    def __init__(self, estimator,
                 phase, 
                 n_jobs, cv_k_fold, parameters,
                 X_train, y_train,
                 X_test, y_test):
        # estimator : ensemble学習器

        # cv : if train : get best parameter
        if phase == "train":
            gscv = GridSearchCV(GradientBoostingClassifier(), 
                                parameters, 
                                verbose = 10, 
                                scoring = "f1",#scoring = "precision" or "recall"
                                n_jobs = n_jobs, cv = cv_k_fold)
            gscv.fit(X_train, y_train)
            best_params = gscv.best_params_
            print "[GBDT's Best Parameter]", gscv.best_params_
            
            clf = GradientBoostingClassifier()
            clf.set_params(**gscv.best_params_)
            del gscv
            clf.fit(X_train, y_train)
            train_loss = clf.train_score_
            test_loss = np.empty(len(clf.estimators_))
            for i, pred in enumerate(clf.staged_predict(X_test)):
                test_loss[i] = clf.loss_(y_test, pred)
            plt.plot(np.arange(len(clf.estimators_)) + 1, test_loss, label='Test')
            plt.plot(np.arange(len(clf.estimators_)) + 1, train_loss, label='Train')
            plt.xlabel('the number of weak learner:Boosting Iterations')
            plt.ylabel('Loss')
            plt.legend(loc="best")
            plt.savefig("loss_cv.png")
            plt.close()
        else:
            best_params = {'loss' : ['deviance'],
                           'learning_rate' : [0.1],
                           'max_depth': [2],
                           'min_samples_leaf': [8],
                           'max_features': [5],#max_features must be in (0, n_features]
                           'max_leaf_nodes' : [20],
                           'subsample' : [0.1],
                           'n_estimators' : [100],
                           'random_state' : [0]}
            
        estimator.set_params(**best_params)
        self.estimator = estimator
        self.one_hot_encoding = None
        
    def fit(self, X, y):
        self.fit_transform(X, y)
        return self
        
    def fit_transform(self, 
                      X, y):
        """
         [estimator]
          <class 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'
          GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
                                     max_depth=2, max_features=None, max_leaf_nodes=2,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=100,
                                     random_state=0, subsample=0.3, verbose=0, warm_start=False)
         [estimator_.estimators_]
          <type 'numpy.ndarray'>
          ensembles of DecisionTreeRegressor
        """
        # 1. learn data structure by gbdt
        self.estimator_ = clone(self.estimator)#Constructs a new estimator
        self.estimator_.fit(X, y)# fit by gbdt with best parameter

        # 2. get trainsformated feature vectors from self.estimator_
        self.binarizers_ = []
        sparse_applications = []
        # --- np.asarray() ---
        # array()と同様．ただし引数がndarrayの場合，コピーでなく引数そのものを返す
        # np.asarray([1,2,3])
        # >array([1, 2, 3])
        # a = np.array([1,2])
        # b = np.asarray(a)# ndarrayを引数とするため，b = aと同値
        # array([1, 2])
        # --- np.ravel() ---
        # It is equivalent to reshape(-1, order=order).
        # x = np.array([[1, 2, 3], [4, 5, 6]])
        # >array([[1, 2, 3],
        # >       [4, 5, 6]])
        # np.ravel(x)
        # >[1 2 3 4 5 6]
        # x.reshape(-1)
        # [1 2 3 4 5 6]>
        estimators = np.asarray(self.estimator_.estimators_).ravel()
        for index, t in enumerate(estimators):
            # for each weak learner
            # t is weak learner
            # DecisionTreeRegressor(criterion=<sklearn.tree._tree.RegressionCriterion object at 0x350d1e0>,
            #                       max_depth=2, 
            #                       max_features=None, 
            #                       max_leaf_nodes=2,
            #                       min_samples_leaf=1, 
            #                       min_samples_split=2,
            #                       min_weight_fraction_leaf=0.0,
            #                       random_state=<mtrand.RandomState object at 0x2ca3b50>,
            #                       splitter=<sklearn.tree._tree.PresortBestSplitter object at 0x2cdcca0>)
            # [Attributes]
            # t.tree_ : object
            # t.max_features_ : int
            # t.feature_importances_ : ndarray
            lb = LabelBinarizer(sparse_output=True)
            sparse_applications.append(lb.fit_transform(t.tree_.apply(X)))
            #print "%d leaves in %d-th tree(weak learner)" % (len(lb.fit_transform(t.tree_.apply(X_train)).toarray().ravel()),
            #                                                 index)
            #print " max_feature_number:%d, sample_size:%d" % (t.max_features_, len(X_train))
            #26049 leaves in 88-th tree(weak learner)
            # max_feature_number:14, sample_size:26049

            self.binarizers_.append(lb) # add tree as weak learner

        self.one_hot_encoding = hstack(sparse_applications)
        
    def transform(self, X, y=None):
        sparse_applications = []
        estimators = np.asarray(self.estimator_.estimators_).ravel() # estimators are ensamble of decision trees
        for t, lb in zip(estimators, self.binarizers_):
            sparse_applications.append(lb.transform(t.tree_.apply(X)))
        return hstack(sparse_applications)

def get_prob(clf, X):
    # [reference]
    # Probability Calinration Curves in sklearn
    if hasattr(clf, "predict_prob"):
        prob_pos = clf.predict_proba(X)[:,1]
    else: # use decision function
        prob_pos = clf.decision_function(X)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min()) # scailing [0,1]
        
    return prob_pos
