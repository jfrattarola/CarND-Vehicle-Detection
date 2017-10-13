#Feature extraction methods from project training lessons
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import glob
from parameters import *

def convert_color(image, color_space=COLOR_SPACE):
    if color_space == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: return np.copy(img)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,  feature_vec=True):
    features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                              visualise=True, feature_vector=False)
    if vis == True:
        return features, hog_image
    return features

def bin_spatial(img, size=SPATIAL_SIZE):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=HIST_BINS):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features(imgs,
                     color_space=COLOR_SPACE,
                     orient=ORIENT,
                     pix_per_cell=PIX_PER_CELL,
                     cell_per_block=CELL_PER_BLOCK,
                     hog_channel=HOG_CHANNEL,
                     spatial_size=SPATIAL_SIZE,
                     hist_bins=HIST_BINS):
    # Create a list to append feature vectors to
    features = [] 
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, color_space)
        feature_ch = feature_image[:,:,0]
        
        #spatial features
        spatial_features = bin_spatial(feature_image, size=spatial_size)

        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)

        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = get_hog_features(feature_ch, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()

        # Append the new feature vector to the features list
        
        features.append(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
    # Return list of feature vectors
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()

    images = glob.glob('{}/vehicles/MiddleClose/image000*.png'.format(FLAGS.dir))
    features = extract_features(images)
    print('features: {}'.format(features[0].shape))
