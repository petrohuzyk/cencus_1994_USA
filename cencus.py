import pandas as pd 
import numpy as np 
import os

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

print("Reading files: \n{} \n{}".format(os.path.join(path, file1), os.path.join(path, file2)))
df_train = pd.read_csv(os.path.join(path, file1), names=names)
df_test = pd.read_csv(os.path.join(path, file2), names=names)
print("File was loaded...")

print("Train data contains {} samples and {} features".format(df_train.shape[0], df_train.shape[1]))
print("Test data contains {} samples and {} features".format(df_test.shape[0], df_test.shape[1]))

data = df_train[['age', 'workclass', 'education', 'gender', 'hours-per-week',
'occupation', 'income']]
# print(data.head())
print(data.gender.value_counts())
# get_dummies will encode only objects
# int (continues) values it isn't encode
print(data.dtypes)

print("Original data features: {}\n{}".format(len(data.columns), data.columns))
data_dummies = pd.get_dummies(data)
print("Dummies data features: {}\n{}".format(len(data_dummies.columns), data_dummies.columns))

features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print("Shape of X:\n{}\nShape of y:\n{}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)
print("Train score: {:.4f}".format(logreg.score(X_train, y_train)))
print("Test score: {:.4f}".format(logreg.score(X_test, y_test)))