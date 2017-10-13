from parameters import *
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from sliding import find_cars
import glob
import sys
import argparse

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[{}] {}{} ...{}\r'.format(bar, percents, '%', suffix))
    sys.stdout.flush()

class BoxQueue ():
    def __init__ (self, maxlen=12):
        self.maxlen = maxlen
        self.bboxes = []

    def put(self, bboxes):
        if (len(self.boxes) > self.maxlen):
            self.boxes.pop(0)
        self.bboxes.append(bboxes)
        
    def get(self):
        res = []
        for bboxes in self.bboxes:
            res.extend(bboxes)
        return res

queue = BoxQueue(NUM_FRAMES)
    
def process_image (image):
    
    res = np.copy (image)
    res = res.astype(np.float32)/255
    
    bboxes, image_bboxes = get_bboxes (image)
    queue.put(bboxes)
    bboxes = queue.get()
    
    final_boxes = get_final_boxes (bboxes, NUM_FRAMES*2)
    res = 255 * draw_boxes(res, final_boxes, color=DEFAULT_BOX_COLOR, thickness=DEFAULT_BOX_THICKNESS)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='project_video.mp4',
                        help='video file to use')
    FLAGS, unparsed = parser.parse_known_args()

    clip = VideoFileClip(FLAGS.video)
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
        draw_img = process_image(img)
        mpimg.imsave('frames/test{}_detections.png'.format(idx+1), draw_img)
