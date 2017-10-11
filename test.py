import pickle
import matplotlib.image as mpimg
from features import extract_features
from training_data import extract_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

if __name__ == '__main__':
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    features_car = get_features('vehicles/Far/image0004.png')
    features_noncar = get_features('non-vehicles/MiddleClose/image0001.png')

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

    X_train, y_train, X_test, y_test = get_data()
    scaled_X_test = X_scaler.transform(X_test)
    predictions = clf.predict(scaled_X_test)
    print('Accuracy on Test Set: {:.2f}%'.format(accuracy_score(y_test, predictions)))

