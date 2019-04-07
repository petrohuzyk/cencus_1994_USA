import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split

def preprocess():
    """
    Reading datasets
    """
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

    df_test = df_test.replace({'income': (r'>.*', r'<.*')}, {'income': ('>50K', '<=50K')}, regex=True)

    dataset = df_train.merge(df_test, how='outer')
    
    return dataset

def remov_attr_dataset():
    '''
    Dataset with removing attributes
    '''
    features = ['age', 'workclass', 'education', 'education-num', 'occupation', 'relationship', 'race', 'gender',
         'capital-gain', 'capital-loss', 'hours-per-week', 'income']
    dataset = preprocess()
    dataset = dataset[features]
    dataset_dummies = pd.get_dummies(dataset)
    X = dataset_dummies.loc[:, 'age':'gender_ Male']
    y = dataset_dummies['income_ >50K']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

def binning_dataset():
    '''
    Dataset with binning attributes
    '''
    dataset_binning = preprocess()

    bins_hours = np.linspace(min(dataset_binning['hours-per-week']), max(dataset_binning['hours-per-week']), 10)
    group_names = ["Very Low", "Low", "Above Low", "Bellow Avg", "Avg", "Above Avg", "Bellow High", "High", "Above High"]
    dataset_binning["hours_binned"] = pd.cut(dataset_binning['hours-per-week'], bins_hours, labels=group_names)

    bins_ages = np.linspace(min(dataset_binning['age']), max(dataset_binning['age']), 8)
    group_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']
    dataset_binning["ages_binned"] = pd.cut(dataset_binning['age'], bins_ages, labels=group_names)

    features = ['ages_binned', 'workclass', 'education', 'education-num', 'occupation', 'relationship', 'race', 'gender',
            'capital-gain', 'capital-loss', 'hours_binned', 'income']
    dataset_binning = dataset_binning[features]
    dataset_dummies = pd.get_dummies(dataset_binning)
    X = dataset_dummies.loc[:, 'education-num':'hours_binned_Above High']
    y = dataset_dummies['income_ >50K']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test