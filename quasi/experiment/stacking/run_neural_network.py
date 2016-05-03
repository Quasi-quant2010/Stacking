# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'
import numpy as np
import cPickle as cp
import theano
from experiment.stacking.experiment_l1 import ExperimentL1
from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.updates import nesterov_momentum
mytrain_x = np.random.randn(100, 10)
print len(mytrain_x)
mytrain_y = np.zeros(100)
mytrain_y[50:] = 1
print len(mytrain_y)
#mytrain_x = mytrain_x.astype(theano.config.floatX)
#mytrain_y = mytrain_y.astype(theano.config.floatX)
net = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('i', layers.InputLayer),
        ('h1', layers.DenseLayer),
        ('h2', layers.DenseLayer),
        ('o', layers.DenseLayer),
        ],
    # layer parameters:
    i_shape=(None, 307),  # 96x96 input pixels per batch
    h1_num_units=100,  # number of units in hidden layer
    h2_num_units=100,  # number of units in hidden layer
    o_nonlinearity=None,  # output layer uses identity function
    o_num_units=1,   #

    # optimization method:
    #update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=1,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1
)
exp = ExperimentL1(train_fname='standard_train.csv', test_fname='standard_test.csv')
#cp.dump((exp.train_x, exp.train_y), open('train_test_temp.pkl', 'wb'), protocol=2)
net.fit(np.asarray(exp.train_x), np.asarray(exp.train_y))
#net.fit(mytrain_x, mytrain_y)



