import pandas as pd 
import numpy as np 
import os

def preprocess():
    path = os.path.join(os.getcwd(), r"Representing Data/Cencus_1994_USA/data")
    file1 = "adult.data"
    file2 = "adult.test"

    names = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'gender',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income',
    ]

    dtype = {
        'age':int,
        'workclass':object,
        'fnlwgt':int,
        'education':object,
        'education-num':int,
        'marital-status':object,
        'occupation':object,
        'relationship':object,
        'race':object,
        'gender':object,
        'capital-gain':int,
        'capital-loss':int,
        'hours-per-week':int,
        'native-country':object,
        'income':str,
    }

    df_train = pd.read_csv(os.path.join(path, file1), names=names, dtype=dtype)
    df_test = pd.read_csv(os.path.join(path, file2), names=names, dtype=dtype)

    data_train = df_train[['age', 'workclass', 'education', 'relationship', 'race', 'gender', 'hours-per-week',
    'occupation', 'native-country', 'marital-status', 'income']]
    data_test = df_test[['age', 'workclass', 'education', 'relationship', 'race', 'gender', 'hours-per-week',
    'occupation', 'native-country', 'marital-status', 'income']]

    data_test = data_test.replace({'income': (r'>.*', r'<.*')}, {'income': ('>50K', '<=50K')}, regex=True)

    data_dummies_train = pd.get_dummies(data_train)
    data_dummies_test = pd.get_dummies(data_test)

    for col in data_dummies_train.columns:
        if not col in data_dummies_test.columns:
            col_loc = data_dummies_train.columns.get_loc(col)
            data_dummies_test.insert(col_loc, col, 0, allow_duplicates=False)
            pass

    features_train = data_dummies_train.loc[:, 'age':'marital-status_ Widowed']
    features_test = data_dummies_test.loc[:, 'age':'marital-status_ Widowed']
    X_train = features_train.values
    y_train = data_dummies_train['income_ >50K'].values
    X_test = features_test.values
    y_test = data_dummies_test['income_ >50K'].values

    return X_train, X_test, y_train, y_test