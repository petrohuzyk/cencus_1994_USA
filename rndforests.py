from sklearn.ensemble import RandomForestClassifier
from data_preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess()
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=12).fit(X_train, y_train)
print("Train score: {:.4f}".format(clf.score(X_train, y_train)))
print("Test score: {:.4f}".format(clf.score(X_test, y_test)))