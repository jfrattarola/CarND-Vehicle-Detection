from utils import *
from parameters import *
import matplotlib.image as mpimg
import math
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
            hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, 
            hist_feat=HIST_FEAT, hog_feat=HOG_FEAT):
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
    xview[1] = imshape[1] if xview[1] is None else xview[1]
    yview[0] = 0 if yview[0] is None else yview[0]
    yview[1] = imshape[0] if yview[1] is None else yview[1]
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
    return windows
            
def draw_boxes( img, bboxes, color=DEFAULT_BOX_COLOR, thickness=DEFAULT_BOX_THICKNESS):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thickness)
    return imcopy

def draw_labeled_boxes( img, labels, color=DEFAULT_BOX_COLOR, thickness=DEFAULT_BOX_THICKNESS):
    bboxes = []
    for car in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxess.append(bbox)
        cv2.rectangle(img, bbox[0], bbox[1], color, thickness)
    return img, bboxes

def get_windows(image, xviews=WINDOWS['x_limit'], yviews=WINDOWS['y_limit'], window_sizes=WINDOWS['size'], overlaps=WINDOWS['overlap']):
    windows=[]
    windows.append(slide(image, xview=xviews[0], yview=yviews[0], window_size=window_sizes[0], overlap=overlaps[0]))
    windows.append(slide(image, xview=xviews[1], yview=yviews[1], window_size=window_sizes[1], overlap=overlaps[1]))
    windows.append(slide(image, xview=xviews[2], yview=yviews[2], window_size=window_sizes[2], overlap=overlaps[2]))
    return windows

if __name__ == '__main__':
    image = mpimg.imread('test_images/test1.jpg')
    windows = get_windows(image)
    images=[]
    images.append(draw_boxes(np.copy(image), windows[0], color=(0,0,0), thickness=4))
    images.append(draw_boxes(np.copy(image), windows[1], color=(0,0,0), thickness=4))
    images.append(draw_boxes(np.copy(image), windows[2], color=(0,0,0), thickness=4))
                            
    images[0] = draw_boxes(images[0], [windows[0][5]], color=(0,255,0),thickness=8)
    images[1] = draw_boxes(images[1], [windows[1][11]], color=(0,255,0),thickness=8)
    images[2] = draw_boxes(images[2], [windows[2][7]], color=(0,255,0),thickness=8)
            
    titles=['128 x 128', '96 x 96', '80 x 80']
    show_images_in_table(images, (3,1), fig_size=(20,14), titles=titles)
