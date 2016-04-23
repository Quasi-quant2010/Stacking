# -*- coding:utf-8 -*-
__author__ = 'zhenouyang'
import os
import cPickle as cp
from utils.config_utils import Config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def fix_abnormal_value(df):

    return df

def scale(train, test):

    mm = MinMaxScaler()

    # train
    train_ids = train.ID.values
    targets = train.TARGET.values
    train1 = train.drop(['ID', 'TARGET'], axis = 1)
    train1 = mm.fit_transform(train1)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic[train.columns[i+1]] = train1[:, i]
    train1 = pd.DataFrame(dic)
    #print train1['TARGET'].value_counts()

    # test
    """
    test_ids = test.ID.values
    test1 = test.drop(['ID'], axis=1)
    test1 = mm.transform(test1)
    dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)
    """
    test_ids = test.ID.values
    targets = test.TARGET.values
    test1 = test.drop(['ID', 'TARGET'], axis = 1)
    test1 = mm.fit_transform(test1)
    dic = {'ID': test_ids, 'TARGET' : targets}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)
    #print test1['TARGET'].value_counts()

    return train1, test1

def standard(train, test):

    ss = StandardScaler()
    
    # train
    train_ids = train.ID.values
    targets = train.TARGET.values
    train1 = train.drop(['ID', 'TARGET'], axis = 1)
    train1 = ss.fit_transform(train1)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic[train.columns[i+1]] = train1[:, i]
    train1 = pd.DataFrame(dic)
    #print train1['TARGET'].value_counts()


    # test
    test_ids = test.ID.values
    targets = test.TARGET.values
    #test1 = test.drop(['ID'], axis=1)
    test1 = test.drop(['ID', 'TARGET'], axis = 1)
    test1 = ss.transform(test1)
    dic = {'ID': test_ids, 'TARGET' : targets}
    #dic = {'ID': test_ids}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)
    #print test1['TARGET'].value_counts()

    return train1, test1



def extend_df(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    sum_col_val = df.sum(axis=1)
    max_col_val = df.max(axis=1)
    min_col_val = df.min(axis=1)
    df['mean'] = mean
    df['std'] = std
    df['sum_col_val'] = sum_col_val
    df['max_col_val'] = max_col_val
    df['min_col_val'] = min_col_val

    return df


def pca(train, test, components=100):

    pca = PCA(n_components=components)

    # train
    train_ids = train.ID.values
    targets = train.TARGET.values
    train1 = train.drop(['ID', 'TARGET'], axis = 1)
    train1 = pca.fit_transform(train1)
    dic = {'ID': train_ids, 'TARGET' : targets}
    for i in xrange(train1.shape[1]):
        dic[train.columns[i+1]] = train1[:, i]
    train1 = pd.DataFrame(dic)

    # test
    test_ids = test.ID.values
    targets = test.TARGET.values
    test1 = test.drop(['ID', 'TARGET'], axis = 1)
    test1 = pca.transform(test1)
    dic = {'ID': test_ids, 'TARGET' : targets}
    for i in xrange(test1.shape[1]):
        dic[test.columns[i+1]] = test1[:, i]
    test1 = pd.DataFrame(dic)

    return train1, test1


def main(stack_setting_):

    # for train set
    fname = os.path.join(Config.get_string('data.path'), 
                         stack_setting_['0-Level']['folder'], 
                         stack_setting_['0-Level']['filter']['train'])
    train = pd.read_csv(fname)

    # for test dataset
    fname = os.path.join(Config.get_string('data.path'), 
                         stack_setting_['0-Level']['folder'], 
                         stack_setting_['0-Level']['filter']['test'])
    test = pd.read_csv(fname)

    print("= Stats Summary in train and test set ")
    train1 = extend_df(train.copy())
    test1 = extend_df(test.copy())
    train1.to_csv(os.path.join(Config.get_string('data.path'), 
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['raw_extend']['train']), index=False)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'], 
                              stack_setting_['0-Level']['raw_extend']['test']), index=False)

    print("= Scailing in train and test set ")
    train1, test1  = scale(train, test)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['scaled']['train']), index=False)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['scaled']['test']), index=False)

    train1 = extend_df(train1)
    test1 = extend_df(test1)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['scaled_extend']['train']), index=False)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['scaled_extend']['test']), index=False)

    train1, test1  = standard(train, test)
    train1.to_csv(os.path.join(Config.get_string('data.path'),
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['standard']['train']), index=False)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'], 
                              stack_setting_['0-Level']['standard']['test']), index=False)
    
    train1 = extend_df(train1)
    test1 = extend_df(test1)
    train1.to_csv(os.path.join(Config.get_string('data.path'), 
                               stack_setting_['0-Level']['folder'], 
                               stack_setting_['0-Level']['standard_extend']['train']), index=False)
    test1.to_csv(os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'], 
                              stack_setting_['0-Level']['standard_extend']['test']), index=False)
    
    return True

