# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 15:59:49 2016

@author: amdds
"""

import datetime
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import zipfile
import time
import shutil
from sklearn.metrics import log_loss
import os
from scipy.sparse import vstack

random.seed(2016)

def run_xgb(train, test, features, target, random_state=0):
    eta = 0.1
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 500
    early_stopping_rounds = 50
    test_size = 0.3

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    #now = datetime.datetime.now()
    sub_file = 'XGBoosttemp.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    test_val = test['device_id'].values
    for i in range(len(test_val)):
        str1 = str(test_val[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def map_column(table, f):
    labels = sorted(table[f].unique())
    mappings = dict()
    for i in range(len(labels)):
        mappings[labels[i]] = i
    table = table.replace({f: mappings})
    return table


def read_train_test():
    # Events
    print('Read events...')
    events = pd.read_csv("../Data/events.csv", dtype={'device_id': np.str})
    events['counts'] = events.groupby(['device_id'])['event_id'].transform('count')
    events_small = events[['device_id', 'counts']].drop_duplicates('device_id', keep='first')

    # Phone brand
    print('Read brands...')
    pbd = pd.read_csv("../Data/phone_brand_device_model.csv", dtype={'device_id': np.str})
    pbd.drop_duplicates('device_id', keep='first', inplace=True)
    pbd = map_column(pbd, 'phone_brand')
    pbd = map_column(pbd, 'device_model')

    # Train
    print('Read train...')
    train = pd.read_csv("../Data/gender_age_train.csv", dtype={'device_id': np.str})
    train = map_column(train, 'group')
    train = train.drop(['age'], axis=1)
    #train = train.drop(['gender'], axis=1)
    train = pd.merge(train, pbd, how='left', on='device_id', left_index=True)
    train = pd.merge(train, events_small, how='left', on='device_id', left_index=True)
    train.fillna(-1, inplace=True)
    train = train.set_index('device_id')

    # Test
    print('Read test...')
    test = pd.read_csv("../Data/gender_age_test.csv", dtype={'device_id': np.str})
    test = pd.merge(test, pbd, how='left', on='device_id', left_index=True)
    test = pd.merge(test, events_small, how='left', on='device_id', left_index=True)
    test.fillna(-1, inplace=True)

    # Features
    features = list(test.columns.values)
    features.remove('device_id')

    return train, test, features, events


train, test, features, events= read_train_test()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_xgb(train, test, features, 'group')
print("LS: {}".format(round(score, 5)))
create_submission(score, test, test_prediction)



# creating a column to indicate which is our predicted group
pred = pd.read_csv("XGBoosttemp.csv")
pred = pred.set_index('device_id')
pred['prediction'] = pred.idxmax(axis=1)

# merging with the test data to see if we've made the right predictions
eva = pd.merge(train, pred,how='left', right_index=True, left_index=True)
eva = eva[['gender','group','prediction']]
eva['predicted gender'] = eva['prediction'].str[0]


# comparing actual and predicted gender and group
eva['gender result'] = (eva['gender'] == eva['predicted gender']).astype(int)
eva['group result'] = (eva['group'] == eva['prediction']).astype(int)

# dropping columns that are not needed
eva = eva[['prediction','predicted gender','gender result','group result']]

# changing column names
eva.columns = ['age','gender','gender result','age result']

# merging with events table to get latitude and longitude values
events = events.set_index('device_id')
# first delete all entries in events where lat long is zero or long<70
events = events[(events.latitude != 0) & (events.longitude >70)]

# then merge and drop duplicate entries
final = eva.merge(events, left_index=True, right_index=True, how='inner')
final['device id'] = final.index
final = final.drop_duplicates('device id',keep='first')

# dropping unnecessary columns
final = final[['device id','gender','age','latitude','longitude']]
# writing result
final.to_csv(('../Visualization/datafiles/XGBoost.csv'), index=False)