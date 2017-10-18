
import pandas as pd
import numpy as np

def recon(reg):
    integer = int(np.round((40*reg)**2)) # gives 2364 for our example
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M

def transform_df(train):
    calc_cols = [col for col in train.columns[train.columns.str.contains('calc')]]
    train = train.drop(calc_cols,axis=1)
    return train

def read_train_test():
    test = pd.read_csv('./data/test.csv')
    test['id'] = test['id'].astype(int)
    train = pd.read_csv('./data/train.csv')

    # Remove calc cols
    train = transform_df(train)
    test = transform_df(test)

    # Expand ps_car_03 based on: https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03
    # train['ps_reg_03_a'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
    # train['ps_reg_03_m'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])
    # test['ps_reg_03_a'] = test['ps_reg_03'].apply(lambda x: recon(x)[0])
    # test['ps_reg_03_m'] = test['ps_reg_03'].apply(lambda x: recon(x)[1])

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
