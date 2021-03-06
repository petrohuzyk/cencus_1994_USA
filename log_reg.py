from sklearn.linear_model import LogisticRegression
from data_preprocess import binning_dataset

X_train, X_test, y_train, y_test = binning_dataset()
logreg = LogisticRegression().fit(X_train, y_train)
print("Train score: {:.4f}".format(logreg.score(X_train, y_train)))
print("Test score: {:.4f}".format(logreg.score(X_test, y_test)))