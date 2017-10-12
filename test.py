import pickle
import matplotlib.image as mpimg
from features import extract_features
from data import extract_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import argparse

if __name__ == '__main__':
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    parser = argparse.ArgumentParser()
    parser.add_argument('--car', type=str, default='image0004.png',
                        help='car image')
    parser.add_argument('--non', type=str, default='image0001.png',
                        help='non-car image')
    FLAGS, unparsed = parser.parse_known_args()

    features_car = extract_features([FLAGS.car])
    features_noncar = extract_features([FLAGS.non])

    features_car_scaled = X_scaler.transform(features_car)
    features_noncar_scaled = X_scaler.transform(features_noncar)

    prediction = clf.predict(features_car_scaled)
    if prediction == 1:
        print('Correct prediction of Car')
    else: print('Incorrect prediction of Car')
    prediction = clf.predict(features_noncar_scaled)
    if prediction == 0:
        print('Correct prediction of Non-Car')
    else: print('Incorrect prediction of Non-Car')

    X_train, y_train, X_test, y_test = extract_data()
    scaled_X_test = X_scaler.transform(X_test)
    predictions = clf.predict(scaled_X_test)
    print('Accuracy on Test Set: {:.2f}%'.format(accuracy_score(y_test, predictions)))

