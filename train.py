import numpy as np
from data import extract_data
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

    X_train, y_train, X_test, y_test, scaler = extract_data(FLAGS.dir)

    svc = svm.SVC()

    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('Training took {} seconds and produced an accuracy of {}'.format(t2-t, round(svc.score(X_test, y_test), 3)))

    with open('clf.pkl', 'wb') as fid:
        pickle.dump(svc, fid)
    with open('scaler.pkl', 'wb') as fid:
        pickle.dump(scaler, fid)
