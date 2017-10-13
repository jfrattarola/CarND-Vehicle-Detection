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
    else: return np.copy(image)

def get_color_channel( image, color_space=COLOR_SPACE, channel=2 ):
    cvt = convert_color(image, color_space)
    return cvt[:,:,channel]

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False,  feature_vec=True):
    if vis == True:
        features, hog_image = hog( img, orientations=orient, 
                                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                                   cells_per_block=(cell_per_block, cell_per_block), 
                                   transform_sqrt=True, 
                                   visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=SPATIAL_SIZE):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.concatenate([color1, color2, color3])
                        
def color_hist(img, nbins=HIST_BINS, bins_range=BINS_RANGE):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def extract_features_img(img, 
                         color_space=COLOR_SPACE,
                         orient=ORIENT,
                         pix_per_cell=PIX_PER_CELL,
                         cell_per_block=CELL_PER_BLOCK,
                         hog_channel=HOG_CHANNEL,
                         spatial_size=SPATIAL_SIZE,
                         hist_bins=HIST_BINS,
                         spatial_feat=True,
                         hist_feat=True,
                         hog_feat=True):
    features=[]
    feature_img = convert_color(img, color_space)
    
    if spatial_feat is True:
        spatial_features = bin_spatial( feature_img, size=spatial_size)
        features.append(spatial_features)

    if hist_feat is True:
        hist_features = color_hist( feature_img, nbins=hist_bins)
        features.append(hist_features)

    if hog_feat is True:
        if color_space == 'GRAY' or color_space == 'GREY':
            if len(feature_img.shape) == 3:
                feature_img = cv2.cvtColor(feature_img, cv2.COLOR_RGB2GRAY)
            hog_features = get_hog_features(feature_img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        elif hog_channel == 'ALL':
            for channel in range(feature_img.shape[2]):
                hog_features.extend(get_hog_features(feature_img[:,:,channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)) 
        else:
            hog_features = get_hog_features(feature_img[:,:,hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)

    features.append(hog_features)

    return np.concatenate(features)

def extract_features_imgs(imgs,
                          color_space=COLOR_SPACE,
                          orient=ORIENT,
                          pix_per_cell=PIX_PER_CELL,
                          cell_per_block=CELL_PER_BLOCK,
                          hog_channel=HOG_CHANNEL,
                          spatial_size=SPATIAL_SIZE,
                          hist_bins=HIST_BINS,
                          spatial_feat=True,
                          hist_feat=True,
                          hog_feat=True):
    # Create a list to append feature vectors to
    features = [] 
    # Iterate through the list of images
    for f in imgs:
        # Read in each one by one
        image = mpimg.imread(f)
        file_features = extract_features_img(image, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)
        features.append(file_features)
    # Return list of feature vectors
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()

    images = glob.glob('{}/vehicles/**/image000*.png'.format(FLAGS.dir))
    features = extract_features_imgs(images)
    print('features: {}'.format(features[0].shape))
