# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack


class XgboostModel:

    def __init__(self, model_params, train_params=None, test_params=None):
        self.model_params = model_params
        if train_params:
            self.train_params = train_params
        else:
            self.train_params = {"num_boost_round": 300 }
        print "%s" % (model_params['model_type'])
        self.test_params = test_params
        fname_parts = ['xgb']
        fname_parts.extend(['{0}#{1}'.format(key, val) for key,val in model_params.iteritems()])
        self.model_out_fname = '-'.join(fname_parts)

    def fit(self, X, y):
        """Fit model."""
        dtrain = xgb.DMatrix(X, label=np.asarray(y))
        #bst, loss, ntree = xgb.train(self.model_params, dtrain, num_boost_round=self.train_params['num_boost_round'])
        self.bst = xgb.train(self.model_params, dtrain, num_boost_round=self.train_params['num_boost_round'])
        #self.loss = loss
        #self.ntree = ntree
        #print loss, ntree

    def predict(self, X):
        """Predict using the xgb model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)

    def to_string(self):
        return self.model_out_fname


class SklearnModel:
    def __init__(self, model_params):
        self.model_params = model_params
        self.model_class = model_params['model_type']
        print "%s" % (model_params['model_type'])
        del model_params['model_type']
        fname_parts = [self.model_class.__name__]
        fname_parts.extend(['{0}#{1}'.format(k,v) for k,v in model_params.iteritems()])
        self.model = self.model_class(**self.model_params)
        self.model_out_fname = '-'.join(fname_parts)

    def fit(self, X, y):
        """Fit model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict using the sklearn model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        #return self.model.predict_proba(X)[:, 1]
        return self.get_predict_proba(self.model, X)

    def to_string(self):
        return self.model_out_fname

    def get_predict_proba(self, 
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


class LasagneModel:

    def __init__(self, model_params):
        import lasagne

        self.model_params = model_params
        print "%s" % (model_params['model_type'])
        l_in = lasagne.layers.InputLayer(shape=(None, model_params['in_size']), input_var=None)

        # Apply 20% dropout to the input data:
        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=model_params['in_dropout'])

        # Add a fully-connected layer of 800 units, using the linear rectifier, and
        # initializing weights with Glorot's scheme (which is the default anyway):
        l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=model_params['h_size'],
                nonlinearity=model_params['nonlinearity'],
                W=lasagne.init.GlorotUniform())

        # We'll now add dropout of 50%:
        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=model_params['h_dropout'])

        # Another 800-unit layer:
        l_hid2 = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=model_params['h_size'],
                nonlinearity=model_params['nonlinearity'])

        # 50% dropout again:
        l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=model_params['h_dropout'])

        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        l_out = lasagne.layers.DenseLayer(
                l_hid2_drop, num_units=1,
                nonlinearity=lasagne.nonlinearities.sigmoid)

        # Each layer is linked to its incoming layer(s), so we only need to pass
        # the output layer to give access to a network in Lasagne:
        self.network = l_out

    def fit(self, X, y):
        import lasagne
        import theano
        import theano.tensor as T
        """Fit model."""
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.binary_crossentropy(prediction, T.vector)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(self.network, trainable=1)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, T.vector)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.vector),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([T.vector, T.vector], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([T.vector, T.vector], [test_loss, test_acc])

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(self.model_params['num_epochs']):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))
            pass

    def predict(self, X):
        """Predict using the sklearn model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self.model.predict_proba(X)[:, 1]

    def to_string(self):
        return self.model_out_fname



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
                 X_train_, y_train_, X_test_, y_test_,
                 param_candidate = None,
                 n_jobs_ = 5, k_fold_=5):
        # estimator : ensemble learner
        #  estimator : GBDT or Random Forest

        # cv : if train : get best parameter
        #scoring = "f1",#scoring = "precision" or "recall"
        if param_candidate is None:
            param_candidate = {'loss' : ['deviance'],
                               'learning_rate' : [0.1],
                               'max_depth': [2],
                               'min_samples_leaf': [8],
                               'max_features': [5],#max_features must be in (0, n_features]
                               'max_leaf_nodes' : [20],
                               'subsample' : [0.1],
                               'n_estimators' : [100],
                               'random_state' : [0]}

        # 
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.grid_search import GridSearchCV
        gscv = GridSearchCV(GradientBoostingClassifier(), 
                            param_candidate, 
                            verbose = 10, 
                            scoring = "f1",#scoring = "precision" or "recall"
                            n_jobs = n_jobs_, cv = k_fold_)
        gscv.fit(X_train_, y_train_)
        self.best_params = gscv.best_params_
        print "[GBDT's Best Parameter]", gscv.best_params_
        del gscv
            
        #clf.set_params(**gscv.best_params_)
        #clf.fit(X_train, y_train)
        #train_loss = clf.train_score_
        #test_loss = np.empty(len(clf.estimators_))
        #for i, pred in enumerate(clf.staged_predict(X_test)):
        #    test_loss[i] = clf.loss_(y_test, pred)

        estimator.set_params(**self.best_params)
        self.estimator = estimator
        self.one_hot_encoding = None

    def get_best_param(self):
        return self.best_params
        
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
            # X should be in np.ndarray format, got <class 'pandas.core.frame.DataFrame'>
            sparse_applications.append(lb.transform(t.tree_.apply(X)))
        return hstack(sparse_applications)

