# -*- coding: utf-8 -*-

from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np

class onehot_encode:
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding

    [Reference]
    https://gist.github.com/ramhiser/982ce339d5f8c9a769a0
    """

    def __init__(self):
        self.vec = DictVectorizer()

    def fit(self, X, cols, y=None):
        """
         X : dataframe
         y : No need
        """

        # the 'outtype' keyword is deprecated, use 'orient' instead
        self.vec.fit(X[cols].to_dict(orient='records'))

        return self

    def transform(self, X, cols, y=None):
        
        #vec_data = pd.DataFrame(self.vec.fit_transform(X[cols].to_dict(orient='records')).toarray())
        vec_data = pd.DataFrame(self.vec.transform(X[cols].to_dict(orient='records')).toarray())
        vec_data.columns = self.vec.get_feature_names()
        vec_data.index = X.index
        
        X = X.drop(cols, axis=1)
        X = X.join(vec_data)
        
        return X
