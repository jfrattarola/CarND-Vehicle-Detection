from parameters import *
from features import extract_features
import glob, os
import numpy as np
from sklearn.utils import shuffle
import argparse

def extract_data(path='.'):
    if os.path.exists('X_train.npy'):
        print('Loading numpy cache')
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')

    else:
        print('Loading files from {}/'.format(path))
        cars = glob.glob('{}/vehicles/**/*.png'.format(path))
        non_cars = glob.glob('{}/non-vehicles/**/*.png'.format(path))

        # Compute features and labels for training data
        car_features = None
        non_car_features = None
        cars_len = len(cars)
        non_cars_len = len(non_cars)
        y = np.concatenate([np.ones(cars_len, dtype=np.int), np.zeros(non_cars_len, dtype=np.int)])
        
        print ('extracting features for {} cars'.format(cars_len))
        car_features = extract_features(cars)

        print ('extracting features for {} non-cars'.format(non_cars_len))
        non_car_features = extract_features(non_cars)

        car_features = np.asarray(car_features)
        non_car_features = np.asarray(non_car_features)
#        y = np.asarray(y)
        X = np.squeeze(np.concatenate((car_features, non_car_features)))

        print(len(y), len(X))
        # Shuffle features and labels in a consistent way
        X, y = shuffle(X, y)

        # Split training and test data
        train_portion = 0.75
        tot_training_samples = int(len(X) * train_portion)
        X_train = X[:tot_training_samples]
        y_train = y[:tot_training_samples]
        X_test = X[tot_training_samples:]
        y_test = y[tot_training_samples:]

        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

    print('\nNumber of training samples: {}'.format(X_train.shape[0]))
    print('Number of test samples: {}'.format(X_test.shape[0]))
    print('Number of positive samples: {}'.format(np.sum(y_test) + np.sum(y_train)))

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()
    X_train, y_train, X_test, y_test = extract_data(FLAGS.dir)
