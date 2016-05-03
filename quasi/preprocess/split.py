# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import check_random_state
import pandas as pd

class File_Split:

    """
     fs = File_Split(test_size=0.5)
     fs.__iter__(fname='/home/username/data/sklearn/stacking/input/adult.data.csv',
                 train_fname='/home/username/data/sklearn/stacking/input/adult.data_train.csv',
                 test_fname='/home/username/data/sklearn/stacking/input/adult.data_test.csv')
    """

    def __init__(self, test_size=0.2, random_state=0):
        self.test_size = test_size
        self.random_state = random_state

    def __iter__(self, fname, train_fname, test_fname):
        data = pd.read_csv(fname)
        n = len(data.index)
        n_test = int(np.ceil(self.test_size * n))
        n_train = n - n_test
        rng = check_random_state(self.random_state)
        permutation = rng.permutation(n)
        ind_test = permutation[:n_test]
        ind_train = permutation[n_test:n_test + n_train]

        data.iloc[ind_train].to_csv(train_fname, index = False)
        data.iloc[ind_test].to_csv(test_fname, index = False)
