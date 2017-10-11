import numpy as np
from data import extract_data
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()

    X_train, y_train, X_test, y_test = extract_data(FLAGS.dir)

    X_scaler = StandardScaler()
    X_scaler.fit(X_train)
    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)
    parameters = {'kernel':('linear', 'rbf'), 'gamma':[1e-3], 'C':[1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    clf.fit(scaled_X_train, y_train)

    with open('clf.pkl', 'wb') as fid:
        pickle.dump(clf, fid)
    with open('scaler.pkl', 'wb') as fid:
        pickle.dump(X_scaler, fid)
