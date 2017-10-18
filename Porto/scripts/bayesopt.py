from __future__ import print_function
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'
__author__ = 'tilii: https://kaggle.com/tilii7'

# ZFTurbo defined first 3 features
# tilii added two new features and Bayesian Optimization
# Bayesian Optimization library credit to Fernando Nogueira https://www.kaggle.com/fnogueira
# Also see https://github.com/fmfn/BayesianOptimization
# Some ideas were taken from Mike Pearmain https://www.kaggle.com/mpearmain
# Also see https://github.com/mpearmain/BayesBoost

import pandas as pd
import numpy as np
import re
from sklearn.cross_validation import train_test_split
from sklearn import manifold, metrics
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bayes_opt import BayesianOptimization
import xgboost as xgb

# Instead of running XGBXlassifier, we do xgbCV, so this takes longer
# We capture stderr and stdout using the function below

import contextlib


@contextlib.contextmanager
def capture():
    import sys
    from io import StringIO
    olderr, oldout = sys.stderr, sys.stdout
    try:
        out = [StringIO(), StringIO()]
        sys.stderr, sys.stdout = out
        yield out
    finally:
        sys.stderr, sys.stdout = olderr, oldout
        out[0] = out[0].getvalue().splitlines()
        out[1] = out[1].getvalue().splitlines()


# Define custom metric gini

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred)

# Define which xgbCV parameters are used for grid search
# and specify all xgbCV parameters


def XGBcv(max_depth, gamma, min_child_weight, max_delta_step, subsample,
          colsample_bytree):
    paramt = {
        'gamma': gamma,
        'booster': 'gbtree',
        'max_depth': max_depth.astype(int),
        'eta': .001,
        'tree_method': 'gpu_hist',
        # Use the line below for classification
        'objective' : 'binary:logistic',
        #'objective': 'multi:softprob',
        #'nthread' : 8,
        # DO NOT use the line below when doing classification
        #'num_class': 12,
        'silent': True,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'max_delta_step': max_delta_step.astype(int),
        'seed': 42
    }

    # Use 10-fold validation if you have time to spare
    #folds = 10
    folds = 5
    cv_score = 0

    print(" Search parameters (%d-fold validation):\n %s" % (folds, paramt),file=log_file)
    log_file.flush()

    # Do not optimize the number of boosting rounds, as early stopping will take care of that

#     with capture() as result:
#         xgb.cv(paramt,
#                dtrain,
#                num_boost_round=20000,
#                stratified=True,
#                nfold=folds,
#                verbose_eval=1,
#                early_stopping_rounds=50,
#                #feval=gini_xgb,
#                metrics='auc',
#                #metrics="mlogloss",
#                show_stdv=True)

    res = xgb.cv(paramt,
               dtrain,
               num_boost_round=20000,
               stratified=True,
               nfold=folds,
               #verbose_eval=1,
               early_stopping_rounds=50,
               #feval=gini_xgb,
               metrics='auc',
               #metrics="mlogloss",
               show_stdv=True)
    # print(type(res))
    # print(res)
    print('', file=log_file)
    print(res, file=log_file)
    log_file.flush()

# All relevant things in XGboost output are in stdout, so we screen result[1]
# for a line with "cv-mean". This line signifies the end of output and contains CV values.
# Next we split the line to extract CV values. We also print the whole CV run into file
# In previous XGboost the output was in stderr, in which case we would need result[0]

    # print('', file=log_file)
    # #for line in result[0]:
    # for line in result[1]:
    #     print(line, file=log_file)
    #     if str(line).find('cv-mean') != -1:
    #         cv_score = float(re.split('[|]| |\t|:', line)[2])
    # log_file.flush()

    # The CV metrics function in XGboost can be lots of things. Some of them need to be maximized, like AUC.
    # If the metrics needs to be minimized, e.g, logloss, the return line below should be a negative number
    # as Bayesian Optimizer only knows how to maximize the function

    cv_score = res['test-auc-mean'].max()
    cv_score = 2*cv_score-1

    #boost_rounds = res['test-auc-mean'].argmax()

    #return (-1.0 * cv_score)
    return cv_score


# def map_column(table, f):
#     labels = sorted(table[f].unique())
#     mappings = dict()
#     for i in range(len(labels)):
#         mappings[labels[i]] = i
#     table = table.replace({f: mappings})
#     return table


# def read_train_test():
#     # App events
#     print('\nReading app events...')
#     ape = pd.read_csv('../input/app_events.csv')
#     ape.drop_duplicates('event_id', keep='first', inplace=True)
#     ape.drop(['app_id'], axis=1)

#     # Events
#     print('Reading events...')
#     events = pd.read_csv('../input/events.csv', dtype={'device_id': np.str})
#     events['counts'] = events.groupby(
#         ['device_id'])['event_id'].transform('count')

#     print('Making events features...')
#     # The idea here is to count the number of installed apps using the data
#     # from app_events.csv above. Also to count the number of active apps.
#     events = pd.merge(events, ape, how='left', on='event_id', left_index=True)
#     events['installed'] = events.groupby(
#         ['device_id'])['is_installed'].transform('sum')
#     events['active'] = events.groupby(
#         ['device_id'])['is_active'].transform('sum')

#     # Below is the original events_small table
#     # events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')
#     # And this is the new events_small table with two extra features
#     events_small = events[['device_id', 'counts', 'installed',
#                            'active']].drop_duplicates('device_id',
#                                                       keep='first')

#     # Phone brand
#     print('Reading phone brands...')
#     pbd = pd.read_csv('../input/phone_brand_device_model.csv',
#                       dtype={'device_id': np.str})
#     pbd.drop_duplicates('device_id', keep='first', inplace=True)
#     pbd = map_column(pbd, 'phone_brand')
#     pbd = map_column(pbd, 'device_model')

#     # Train
#     print('Reading train data...')
#     train = pd.read_csv('../input/gender_age_train.csv',
#                         dtype={'device_id': np.str})
#     train = map_column(train, 'group')
#     train = train.drop(['age'], axis=1)
#     train = train.drop(['gender'], axis=1)
#     print('Merging features with train data...')
#     train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
#     train = pd.merge(train,
#                      events_small,
#                      how='left',
#                      on='device_id',
#                      left_index=True)
#     train.fillna(-1, inplace=True)

#     # Test
#     print('Reading test data...')
#     test = pd.read_csv('../input/gender_age_test.csv',
#                        dtype={'device_id': np.str})
#     print('Merging features with test data...\n')
#     test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
#     test = pd.merge(test,
#                     events_small,
#                     how='left',
#                     on='device_id',
#                     left_index=True)
#     test.fillna(-1, inplace=True)

#     # Features
#     features = list(test.columns.values)
#     features.remove('device_id')
#     return train, test, features

def transform_df(train):
    calc_cols = [col for col in train.columns[train.columns.str.contains('calc')]]
    train = train.drop(calc_cols,axis=1)
    return train

def read_train_test():
    test = pd.read_csv('./data/test.csv')
    test['id'] = test['id'].astype(int)
    train = pd.read_csv('./data/train.csv')
    #test = rename_cols(test)
    #train = rename_cols(train)
    train = transform_df(train)
    test = transform_df(test)

    ps_car_09_cat_mappings = {}
    ps_car_09_cat_mappings[4] = 1
    ps_car_09_cat_mappings[3] = 1
    ps_car_09_cat_mappings[2] = 1
    ps_car_09_cat_mappings[1] = 1
    ps_car_09_cat_mappings[0] = 0
    ps_car_09_cat_mappings[-1] = 1
    train['ps_car_09_cat_bin'] = train['ps_car_09_cat'].replace(ps_car_09_cat_mappings)
    test['ps_car_09_cat_bin'] = test['ps_car_09_cat'].replace(ps_car_09_cat_mappings)

    ps_car_07_cat_mappings = {}
    ps_car_07_cat_mappings[1] = 0
    ps_car_07_cat_mappings[0] = 1
    ps_car_07_cat_mappings[-1] = 1
    train['ps_car_07_cat_bin'] = train['ps_car_07_cat'].replace(ps_car_07_cat_mappings)
    test['ps_car_07_cat_bin'] = test['ps_car_07_cat'].replace(ps_car_07_cat_mappings)

    ps_car_05_cat_mappings = {}
    ps_car_05_cat_mappings[1] = 1
    ps_car_05_cat_mappings[0] = 1
    ps_car_05_cat_mappings[-1] = 0
    train['ps_car_05_cat_bin'] = train['ps_car_05_cat'].replace(ps_car_05_cat_mappings)
    test['ps_car_05_cat_bin'] = test['ps_car_05_cat'].replace(ps_car_05_cat_mappings)

    ps_car_03_cat_mappings = {}
    ps_car_03_cat_mappings[1] = 1
    ps_car_03_cat_mappings[0] = 1
    ps_car_03_cat_mappings[-1] = 0
    train['ps_car_03_cat_bin'] = train['ps_car_03_cat'].replace(ps_car_03_cat_mappings)
    test['ps_car_03_cat_bin'] = test['ps_car_03_cat'].replace(ps_car_03_cat_mappings)

    train = train.drop(['ps_car_09_cat','ps_car_07_cat','ps_car_05_cat','ps_car_03_cat'],axis=1)
    test = test.drop(['ps_car_09_cat','ps_car_07_cat','ps_car_05_cat','ps_car_03_cat'],axis=1)

    return train, test

    # mean_mappings = (train.groupby(['ps_reg_03'])['target'].describe()
    #             .unstack()[['mean']] > .05).astype(np.int).unstack().unstack()
    # ps_reg_03_mean_mappings = {}
    # for val in mean_mappings.columns:
    #     ps_reg_03_mean_mappings[val] = mean_mappings[val]['mean']
    # train['ps_reg_03_mean'] = train['ps_reg_03'].replace(ps_reg_03_mean_mappings)
    # test['ps_reg_03_mean'] = test['ps_reg_03'].replace(ps_reg_03_mean_mappings)

    # count_mappings = (train.groupby(['ps_reg_03'])['target'].describe()
    #             .unstack()[['count']] > 50).astype(np.int).unstack().unstack()
    # ps_reg_03_count_mappings = {}
    # for val in count_mappings.columns:
    #     ps_reg_03_count_mappings[val] = count_mappings[val]['count']
    # train['ps_reg_03_count'] = train['ps_reg_03'].replace(ps_reg_03_count_mappings)
    # test['ps_reg_03_count'] = test['ps_reg_03'].replace(ps_reg_03_count_mappings)

    # def reg_03(row):
    #     if row['ps_reg_03_count'] == 1 and row['ps_reg_03_mean'] == 1:
    #         return 1
    #     else:
    #         return 0

    # train['ps_reg_03_bin'] = train.apply(reg_03,axis=1)
    # test['ps_reg_03_bin'] = test.apply(reg_03,axis=1)

    # train = train.drop(['ps_reg_03_mean','ps_reg_03_count','ps_reg_03'],axis=1)
    # test = test.drop(['ps_reg_03_mean','ps_reg_03_count','ps_reg_03'],axis=1)


    # Reference to his functioning script:

# train, test, features = read_train_test()
# print('Length of train: ', len(train))
# print('Length of test: ', len(test))
# print('Features [{}]: {}\n'.format(len(features), sorted(features)))
# train_df = pd.DataFrame(data=train)
# X = train_df.drop(['group', 'device_id'], axis=1).values
# Y = train_df['group'].values
# dtrain = xgb.DMatrix(X, label=Y)

# My addition to the script

train, test = read_train_test()
X = train.drop(['id', 'target'], axis=1).values
Y = train['target'].values
dtrain = xgb.DMatrix(X, label=Y)

# Create a file to store XGBoost output
# New lines are added to this file rather than overwriting it
log_file = open("XGBoost-output-from-BOpt-eta.001.txt", 'a')

# Do hyperparameter search by Bayesian Optimization

# Below are real production parameters
XGB_BOpt = BayesianOptimization(XGBcv, { 'max_depth': (2, 12),
                                         'gamma': (0.0001, 2.0),
                                         'min_child_weight': (1, 10),
                                         'max_delta_step': (0, 5),
                                         'subsample': (0.2, 1.0),
                                         'colsample_bytree' :(0.2, 1.0)})
                                         #'eta': (.3, .001)})

# These aare the tuned parameters once running it a few times unused as of yet, becaus
# I changed to variable eta
# XGB_BOpt = BayesianOptimization(XGBcv, { 'max_depth': (2, 8),
#                                          'gamma': (0.0001, 2.0),
#                                          'min_child_weight': (6, 10),
#                                          'max_delta_step': (0, 5),
#                                          'subsample': (0.5, 1.0),
#                                          'colsample_bytree' :(0.2, 0.7),
#                                          'eta': (.3, .001)})

# These parameters will allow for rapid grid search
# XGB_BOpt = BayesianOptimization(XGBcv, {'max_depth': (4, 6),
#                                         'gamma': (0.0001, 0.005),
#                                         'min_child_weight': (1, 2),
#                                         'max_delta_step': (0, 1),
#                                         'subsample': (0.2, 0.4),
#                                         'colsample_bytree': (0.2, 0.4)})

print('\n', file=log_file)
log_file.flush()

print('Running Bayesian Optimization ...\n')
# XGB_BOpt.maximize(init_points=5, n_iter=10)
# Production parameters
XGB_BOpt.maximize(init_points=5, n_iter=50)

print('\nFinal Results')
print('XGBOOST: %f' % XGB_BOpt.res['max']['max_val'])
print('\nFinal Results', file=log_file)
print('XGBOOST: %f' % XGB_BOpt.res['max']['max_val'], file=log_file)
log_file.flush()
log_file.close()
