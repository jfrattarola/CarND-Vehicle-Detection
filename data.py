from parameters import *
from features import extract_features
import glob, os
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        print ('extracting features for {} cars'.format(len(cars)))
        car_features = extract_features_imgs(cars, color_space=COLOR_SPACE,
                                             spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                                             orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                                             cell_per_block=CELL_PER_BLOCK, 
                                             hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, 
                                             hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
        print ('extracting features for {} non-cars'.format(len(non_cars)))
        non_car_features = extract_features_imgs(non_cars, color_space=COLOR_SPACE,
                                                 spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
                                                 orient=ORIENT, pix_per_cell=PIX_PER_CELL, 
                                                 cell_per_block=CELL_PER_BLOCK, 
                                                 hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, 
                                                 hist_feat=HIST_FEAT, hog_feat=HOG_FEAT)
        
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features), dtype=np.int), np.zeros(len(non_car_features), dtype=np.int)))
        
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25, random_state=rand_state)

        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
        np.save('X_test.npy', X_test)
        np.save('y_test.npy', y_test)

    print('\nNumber of training samples: {}'.format(X_train.shape[0]))
    print('Number of test samples: {}'.format(X_test.shape[0]))
    print('Number of positive samples: {}'.format(np.sum(y_test) + np.sum(y_train)))

    return X_train, y_train, X_test, y_test, X_scaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()
    X_train, y_train, X_test, y_test, _ = extract_data(FLAGS.dir)
