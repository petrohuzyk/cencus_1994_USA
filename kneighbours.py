from sklearn.neighbors import KNeighborsClassifier
from data_preprocess import binning_dataset

X_train, X_test, y_train, y_test = binning_dataset()
clf = KNeighborsClassifier(n_neighbors=14).fit(X_train, y_train)
print("Train score: {:.4f}".format(clf.score(X_train, y_train)))
print("Test score: {:.4f}".format(clf.score(X_test, y_test)))