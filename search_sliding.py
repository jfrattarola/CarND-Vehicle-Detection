from parameters import *
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from features import extract_features_img
import glob

#given windows, return list of positives
def search( img, windows, clf, X_scaler, color_space=COLOR_SPACE, 
            spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
            hist_range=BINS_RANGE, orient=ORIENT, 
            pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
            hog_channel=HOG_CHANNEL, spatial_feat=True, 
            hist_feat=True, hog_feat=True):
    positives=[]
    for window in windows:
        img_tosearch = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], 
                                  (64,64), interpolation=cv2.INTER_AREA)
        features = extract_features_img(img_tosearch, color_space=color_space, 
                                        spatial_size=spatial_size, hist_bins=hist_bins, 
                                        orient=orient, pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = X_scaler.transform(np.array(features).reshape(1, -1))        
        pred = clf.predict(test_features)
        if pred == 1:
            positives.append(window)
    return positives

        
# Define a single function that can extract features using hog sub-sampling and make predictions
def slide(img, xview=[None, None], yview=[None, None], window_size=(64,64), overlap=(0.5, 0.5)):
    imshape = img.shape
    xview[0] = 0 if xview[0] is None else xview[0]
    xview[1] = 0 if xview[1] is None else xview[1]
    yview[0] = 0 if yview[0] is None else yview[0]
    yview[1] = 0 if yview[1] is None else yview[1]
    sizey = yview[1] - yview[0]
    sizex = xview[1] - xview[0]
    stepy = int(window_size[0] * overlap[0])
    stepx = int(window_size[1] * overlap[1])
    nysteps = int(math.floor(1.0 * sizey / stepy)) - 1
    nxsteps = int(math.floor(1.0 * sizex / stepx)) - 1

    windows = []
    for y in range(nysteps):
        for x in range(nxsteps):
            xleft = xview[0] + x*stepx
            ytop = yview[0] + y*stepy
            window = ((xleft, ytop), (xleft + window_size[1], ytop + window_size[0]))
            windows.append(window)

def draw_boxes( img, bboxes, color=DEFAULT_BOX_COLOR, thickness=DEFAULT_BOX_THICKNESS):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thickness)
    returne imcopy

if __name__ == '__main__':
    pass
"""
    # load classifier
    print('Loading Classifier')
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    print('Loading Scaler')
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    test_images = glob.glob('test_images/*jpg')
    for idx, image in enumerate(test_images):
        print('Looking for cars in test image: test_images/test{}.jpg'.format(idx+1))
        img = mpimg.imread(image)
        out_img, _ = find_cars(img, YSTART, YSTOP, SCALES, clf, X_scaler, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, COLOR_SPACE, HOG_CHANNEL)
        mpimg.imsave('output_images/test{}_sliding_window_test.png'.format(idx+1), out_img)
"""
