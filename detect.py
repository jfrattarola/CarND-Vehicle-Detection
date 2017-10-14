from parameters import *
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from heat import *
from search_sliding import *
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
        if (len(self.bboxes) > self.maxlen):
            self.bboxes.pop(0)
        self.bboxes.append(bboxes)
        
    def get(self):
        res = []
        for bboxes in self.bboxes:
            res.extend(bboxes)
        return res
    
    def process_image (self, image, settings, num, clf, X_scaler):
        bboxes, _ = get_bboxes (image, settings, num, clf, X_scaler)
        self.put(bboxes)
        bboxes = self.get()
    
        final_boxes = get_final_boxes (bboxes, NUM_FRAMES*2.5)
        res = draw_boxes(res, final_boxes, color=DEFAULT_BOX_COLOR, thickness=DEFAULT_BOX_THICKNESS)
        
        return res

class Detector():
    def __init__(self, maxlen=15):
        self.prev=[]
        self.maxlen = maxlen

    def add(self, bboxes):
        if len(bboxes) > 0:
            self.prev.extend(bboxes)
            l = len(self.prev)
            while(l > self.maxlen):
                self.prev.pop(0)# = self.prev[l-self.maxlen]
                l-=1
        
    def process_image (self, image, settings, num, clf, X_scaler):
        bboxes, _ = get_bboxes (image, settings, num, clf, X_scaler)
        self.add(bboxes)
        heatmap_image = np.zeros_like(image[:,:,0])
        for bbox in self.prev:
            try :
                heatmap_image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
            except:
                print("ERROR: {}".format(bbox))
        heatmap_image, labels = threshold_and_label(heatmap_image, 1 + len(self.prev)//2)
        draw_image, _ = draw_labeled_bboxes(np.copy(image), labels)
        return draw_image
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='project_video.mp4',
                        help='video file to use')
    FLAGS, unparsed = parser.parse_known_args()

    print('Will detect cars from video {}'.format(FLAGS.video))
    clip = VideoFileClip(FLAGS.video)
    frames = int(clip.fps * clip.duration)
    image_folder = "frames/"

    # load classifier
    print('Loading Classifier')
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    print('Loading Scaler')
    # load scaler
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

#    queue = BoxQueue(NUM_FRAMES)
    detector = Detector(NUM_FRAMES)

    print('Processing video...')
    for idx, img in enumerate(clip.iter_frames()):
        progress(idx+1, frames)
#        draw_img = queue.process_image(img, WINDOWS, 3, clf, X_scaler)
        draw_img = detector.process_image(img, WINDOWS, 3, clf, X_scaler)
        mpimg.imsave('frames/test{:04d}_detections.png'.format(idx+1), draw_img)
