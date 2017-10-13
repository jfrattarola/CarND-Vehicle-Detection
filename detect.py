from parameters import *
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from sliding import find_cars
from heat import *
import glob
import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

if __name__ == '__main__':
    clip = VideoFileClip("project_video.mp4")
    frames = int(clip.fps * clip.duration)
    image_folder = "frames/"
    video_file = 'processed_video.mp4'

    # load classifier
    print('Loading Classifier')
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    print('Loading Scaler')
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    print('Processing video...')
    for idx, img in enumerate(clip.iter_frames()):
        progress(idx+1, frames)
        _, box_list = find_cars(img, YSTART, YSTOP, SCALES, clf, X_scaler, orient, PIX_PER_CELL, CELL_PER_BLOCK, SPATIAL_SIZE, HIST_BINS, COLOR_SPACE)

        if idx == 0:
            heat = np.zeros_like(img[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = add_heat(heat, box_list)
        heat = remove_old_heat(heat)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, THRESHOLD)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        mpimg.imsave('frames/test{}_detections.png'.format(idx+1), draw_img)
