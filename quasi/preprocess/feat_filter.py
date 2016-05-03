# -*- coding: utf-8 -*-


import os
import itertools
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from utils.config_utils import Config
from preprocess.one_hot_encoding import onehot_encode


def remove_feat_constants(data_frame):
    """
    Remove feature vectors containing one unique value, because such features do not have predictive value.
    :param data_frame:
    :return:
    """
    print("Deleting zero variance features...")
    # Let's get the zero variance features by fitting VarianceThreshold
    # selector to the data, but let's not transform the data with
    # the selector because it will also transform our Pandas data frame into
    # NumPy array and we would like to keep the Pandas data frame. Therefore,
    # let's delete the zero variance features manually.
    n_features_originally = data_frame.shape[1]
    selector = VarianceThreshold()
    selector.fit(data_frame)
    # Get the indices of zero variance feats
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(data_frame.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
    # Delete zero variance feats from the original pandas data frame
    feat_deleted = [name for name in data_frame.columns[feat_ix_delete]]
    data_frame = data_frame.drop(labels=feat_deleted,
                                 axis=1)
    # Print info
    n_features_deleted = feat_ix_delete.size
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame, feat_deleted


def remove_feat_identicals(data_frame):
    """
    Find feature vectors having the same values in the same order and remove all but one of those redundant features.
    :param data_frame:
    :return:
    """
    print("Deleting identical features...")
    n_features_originally = data_frame.shape[1]
    # Find the names of identical features by going through all the
    # combinations of features (each pair is compared only once).
    feat_names_delete = []
    for feat_1, feat_2 in itertools.combinations(
            iterable=data_frame.columns, r=2):
        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):
            feat_names_delete.append(feat_2)
    feat_names_delete = np.unique(feat_names_delete)
    # Delete the identical features
    data_frame = data_frame.drop(labels=feat_names_delete, axis=1)
    n_features_deleted = len(feat_names_delete)
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame, feat_names_delete


def main(stack_setting_):

    """
     [rawdata2filterdata Step]
      1. Reading raw datasets
      2. Droping useless feat columns in training set
      3. Droping useless feat columns in test set
    """

    raw_train_path = os.path.join(Config.get_string('data.path'), 
                                  stack_setting_['0-Level']['folder'],
                                  stack_setting_['0-Level']['raw']['train'])
    raw_test_path = os.path.join(Config.get_string('data.path'), 
                                 stack_setting_['0-Level']['folder'],
                                 stack_setting_['0-Level']['raw']['test'])
    print("= Reading raw datasets ...")

    names = ("age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, TARGET").split(', ')

    raw_train = pd.read_csv(raw_train_path, names=names, skiprows=1)#, index_col=0, sep=','
    #raw_trai['TARGET'] = (raw_trai['TARGET'].values == ' >50K').astype(np.int32)
    #raw_train = raw_train.apply(lambda x: pd.factorize(x)[0])
    train_path = os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'],
                              stack_setting_['0-Level']['train'])
    raw_train.to_csv(train_path, index=True, index_label='ID')


    raw_test = pd.read_csv(raw_test_path, names=names, skiprows=1)#, index_col=0, sep=','
    #raw_test['TARGET'] = (raw_test['TARGET'].values == ' >50K').astype(np.int32)
    #raw_test = raw_test.apply(lambda x: pd.factorize(x)[0])
    test_path = os.path.join(Config.get_string('data.path'), 
                             stack_setting_['0-Level']['folder'],
                             stack_setting_['0-Level']['test'])
    raw_test.to_csv(test_path, index=True, index_label='ID')



def main_transform(stack_setting_):

    """
     [rawdata2filterdata Step]
      1. Reading raw datasets
      2. Droping useless feat columns in training set
      3. Droping useless feat columns in test set
    """

    raw_train_path = os.path.join(Config.get_string('data.path'), 
                                  stack_setting_['0-Level']['folder'],
                                  stack_setting_['0-Level']['raw']['train'])
    raw_test_path = os.path.join(Config.get_string('data.path'), 
                                 stack_setting_['0-Level']['folder'],
                                 stack_setting_['0-Level']['raw']['test'])
    print("= Reading raw datasets ...")

    names = ("age, workclass, fnlwgt, education, education-num, material, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, country, TARGET").split(', ')

    # train
    raw_train = pd.read_csv(raw_train_path, names=names, skiprows=1)#, index_col=0, sep=','
    train_labels = pd.DataFrame((raw_train['TARGET'].values == ' >50K').astype(np.int32), 
                                columns=['TARGET'])
    raw_train = raw_train.drop(labels=['TARGET'], axis=1)
    #print 'train summary'
    #print raw_train[['age', 'hours-per-week', 'fnlwgt']].describe()
    for key in ['fnlwgt', 'capital-gain', 'capital-loss']:
        raw_train['%s_%s' % (key, 'log')] = np.log(raw_train[key] + 1.0).astype(np.float32)
        raw_train = raw_train.drop(labels=[key], axis=1)
    raw_train['TARGET'] = train_labels; del train_labels

    # test
    raw_test = pd.read_csv(raw_test_path, names=names, skiprows=1)#, index_col=0, sep=','
    test_labels = pd.DataFrame((raw_test['TARGET'].values == ' >50K').astype(np.int32), 
                                columns=['TARGET'])
    raw_test = raw_test.drop(labels=['TARGET'], axis=1)
    print 'test summary-1'
    print raw_test.describe()
    for key in ['fnlwgt', 'capital-gain', 'capital-loss']:
        raw_test['%s_%s' % (key, 'log')] = np.log(raw_test[key] + 1.0).astype(np.float32)
        raw_test = raw_test.drop(labels=[key], axis=1)
    print '\n'
    print 'test summary-2'
    print raw_test.describe()
    raw_test['TARGET'] = test_labels; del test_labels


    # main
    print("= Transform Categorical Variables into One-Hot-Encoding ...")

    categoricals = ['workclass',  'education',  'material', 'occupation', 'relationship', 'race', 'sex', 'country']
    combined = pd.concat([raw_train, raw_test])
    #print combined.head()
    #combined_categoricals = combined[categoricals]

    clf = onehot_encode()
    clf.fit(combined, categoricals)
    train_path = os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'],
                              stack_setting_['0-Level']['train'])

    # dump train
    raw_train = clf.transform(raw_train, categoricals)
    train_label = raw_train.TARGET.values
    raw_train = raw_train.drop(labels=['TARGET'], axis=1)
    raw_train['TARGET'] = pd.DataFrame(train_label); del train_label
    train_path = os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'],
                              stack_setting_['0-Level']['train'])
    raw_train.to_csv(train_path, index=True, index_label='ID')

    # dump test
    raw_test = clf.transform(raw_test, categoricals)
    test_label = raw_test.TARGET.values
    raw_test = raw_test.drop(labels=['TARGET'], axis=1)
    raw_test['TARGET'] = pd.DataFrame(test_label); del test_label
    test_path = os.path.join(Config.get_string('data.path'), 
                              stack_setting_['0-Level']['folder'],
                              stack_setting_['0-Level']['test'])
    raw_test.to_csv(test_path, index=True, index_label='ID')
