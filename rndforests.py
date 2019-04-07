from sklearn.ensemble import RandomForestClassifier
from data_preprocess import remov_attr_dataset

X_train, X_test, y_train, y_test = remov_attr_dataset()
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=16).fit(X_train, y_train)
print("Train score: {:.4f}".format(clf.score(X_train, y_train)))
print("Test score: {:.4f}".format(clf.score(X_test, y_test)))