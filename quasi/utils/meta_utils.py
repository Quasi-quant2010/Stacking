# -*- coding: utf-8 -*-

import os
import sys
import json
from subprocess import Popen, PIPE

import pandas as pd

from utils.config_utils import Config

from preprocess.split import File_Split


def combine_meta_features(stack_setting_):

    #data_folder = stack_setting_['1-Level' ]['meta_features']
    #fname = stack_setting_['setting']['name']
    #fname = os.path.join(Config.get_string('data.path'), data_folder, fname)

    train_merge = []
    test_merge = []
    for model_name in stack_setting_['1-Level'].keys():
        try:
            if model_name == 'gbdt_linear':
                # train
                folder = stack_setting_['1-Level'][model_name]['lower']['meta_feature']['folder']
                train_fname = stack_setting_['1-Level'][model_name]['lower']['meta_feature']['train']
                cmd = "ls %s%s/%s*%s*.%s" % (Config.get_string('data.path'),
                                             folder,
                                             "_".join(".".join(train_fname.split('.')[:-1]).split("_")[:-1]), 
                                             ".".join(train_fname.split('.')[:-1]).split("_")[-1], 
                                             train_fname.split('.')[-1])
                p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
                for line in iter(p.stdout.readline, b''):
                    train = pd.read_csv(line.rstrip('\n'))
                    col_name = train.columns.values[:-1]
                    X_train = train[col_name]
                    col_name = train.columns.values[-1]
                    y_train = train[col_name]
                    train_merge.append(X_train)

                # test
                folder = stack_setting_['1-Level'][model_name]['lower']['meta_feature']['folder']
                test_fname = stack_setting_['1-Level'][model_name]['lower']['meta_feature']['test']
                cmd = "ls %s%s/%s*%s*.%s" % (Config.get_string('data.path'),
                                             folder,
                                             "_".join(".".join(test_fname.split('.')[:-1]).split("_")[:-1]), 
                                             ".".join(test_fname.split('.')[:-1]).split("_")[-1], 
                                             test_fname.split('.')[-1])
                p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
                for line in iter(p.stdout.readline, b''):
                    test = pd.read_csv(line.rstrip('\n'))
                    col_name = test.columns.values[:-1]
                    X_test = test[col_name]
                    col_name = test.columns.values[-1]
                    y_test = test[col_name]
                    test_merge.append(X_test)
            else:
                # train
                folder = stack_setting_['1-Level'][model_name]['meta_feature']['folder']
                train_fname = stack_setting_['1-Level'][model_name]['meta_feature']['train']
                train_fname = os.path.join(Config.get_string('data.path'), folder, train_fname)
                #data1.columns.values
                train = pd.read_csv(train_fname)
                col_name = train.columns.values[:-1]
                X_train = train[col_name]
                col_name = train.columns.values[-1]
                y_train = train[col_name]
                train_merge.append(X_train)

                # test
                folder = stack_setting_['1-Level'][model_name]['meta_feature']['folder']
                test_fname = stack_setting_['1-Level'][model_name]['meta_feature']['test']
                test_fname = os.path.join(Config.get_string('data.path'), folder, test_fname)
                #data1.columns.values
                test = pd.read_csv(test_fname)
                col_name = test.columns.values[:-1]
                X_test = test[col_name]
                col_name = test.columns.values[-1]
                y_test = test[col_name]
                test_merge.append(X_test)

        except:
            pass

    

    train_merge.append(y_train)
    train_merge = pd.concat(train_merge, ignore_index=False, axis=1)
    #print train_merge.head(10)
    folder = stack_setting_['1-Level']['meta_features']['folder']
    train_fname = stack_setting_['1-Level']['meta_features']['train']
    train_fname = os.path.join(Config.get_string('data.path'), folder, train_fname)
    train_merge.to_csv(train_fname, index=False)

    test_merge.append(y_test)
    test_merge = pd.concat(test_merge, ignore_index=False, axis=1)
    #print test_merge.head(10)
    folder = stack_setting_['1-Level']['meta_features']['folder']
    test_fname = stack_setting_['1-Level']['meta_features']['test']
    test_fname = os.path.join(Config.get_string('data.path'), folder, test_fname)
    test_merge.to_csv(test_fname, index=False)
    return True


def make_hold_out(stack_setting_):
    """
     input
      train
     output
      train, ptrain, ptest
    """

    split_ratio = stack_setting_['2-Level']['blending']['hold_out_ratio']

    # train
    folder = stack_setting_['1-Level']['meta_features']['folder']
    train_fname = stack_setting_['1-Level']['meta_features']['train']
    train_fname = os.path.join(Config.get_string('data.path'), folder, train_fname)

    # 
    meta_train_at_blend = os.path.join(Config.get_string('data.path'),
                                       stack_setting_['2-Level']['blending']['folder'],
                                       stack_setting_['2-Level']['blending']['train'])
    meta_ptrain_at_blend = os.path.join(Config.get_string('data.path'),
                                        stack_setting_['2-Level']['blending']['folder'],
                                        stack_setting_['2-Level']['blending']['ptrain'])
    meta_ptest_at_blend = os.path.join(Config.get_string('data.path'),
                                       stack_setting_['2-Level']['blending']['folder'],
                                       stack_setting_['2-Level']['blending']['ptest'])
    meta_test_at_blend = os.path.join(Config.get_string('data.path'),
                                      stack_setting_['2-Level']['blending']['folder'],
                                      'meta_test_at_blend.csv')

    # 1. split train file into train and hold out 
    fs = File_Split(test_size=split_ratio)
    fs.__iter__(fname = train_fname,
                train_fname = meta_train_at_blend,
                test_fname = meta_test_at_blend)
    del fs

    # 2. split 
    fs = File_Split(test_size=0.5)
    fs.__iter__(fname = meta_test_at_blend,
                train_fname = meta_ptrain_at_blend,
                test_fname = meta_ptest_at_blend)
    del fs

    return True

def make_hold_out_backup(stack_setting_):
    """
     input
      train
     output
      train, ptrain, ptest
    """

    split_ratio = stack_setting_['2-Level']['blending']['hold_out_ratio']

    # train
    folder = stack_setting_['1-Level']['meta_features']['folder']
    train_fname = stack_setting_['1-Level']['meta_features']['train']
    train_fname = os.path.join(Config.get_string('data.path'), folder, train_fname)
    train = pd.read_csv(train_fname)

    nrows = len(train.index)
    #a,b = int(nrows * split_ratio), nrows - int(nrows * split_ratio)
    a = nrows - int(nrows * split_ratio)
    train, hold_out = train[:a], train[a:]

    # train data for(meta_feature, label)
    train.to_csv(os.path.join(Config.get_string('data.path'),
                              stack_setting_['2-Level']['blending']['folder'],
                              stack_setting_['2-Level']['blending']['train']), index=False)


    nrows = len(hold_out.index)
    a = int(nrows * 0.5) # for hold out set, we split half train and test data set.
    p_train, p_test = hold_out[:a], hold_out[a:]
    p_train.to_csv(os.path.join(Config.get_string('data.path'),
                                stack_setting_['2-Level']['blending']['folder'],
                                stack_setting_['2-Level']['blending']['ptrain']), index=False)
    p_test.to_csv(os.path.join(Config.get_string('data.path'),
                               stack_setting_['2-Level']['blending']['folder'],
                               stack_setting_['2-Level']['blending']['ptest']), index=False)

    print '----------- train data -----------'
    print train['label'].value_counts()
    print '----------- p_train_data -----------'
    print p_train['label'].value_counts()
    print '----------- p_test_data -----------'
    print p_test['label'].value_counts()


    return True
