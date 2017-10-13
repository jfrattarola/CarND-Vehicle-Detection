from parameters import *
from utils import *
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from search_sliding import search, slide
import glob

def get_bboxes(img, settings, num, clf, X_scaler, color_space=COLOR_SPACE, 
               spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
               hist_range=BINS_RANGE, orient=ORIENT, 
               pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
               hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, 
               hist_feat=HIST_FEAT, hog_feat=HOG_FEAT):
    res_img = np.copy(img)
    bboxes=[]

    for i in range(num):
        windows = slide(res_img, settings['x_limit'][i], settings['y_limit'][i], settings['size'][i], settings['overlap'][i])
        hot = search(img, windows, clf, X_scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        bboxes.extend(hot)
        res_img = draw_boxes(res_img, hot, color=(0,0,1), thickness=4)

    return bboxes, res_img

if __name__ == '__main__':
    
