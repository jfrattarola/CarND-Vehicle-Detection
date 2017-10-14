from parameters import *
from utils import *
from search_sliding import draw_boxes
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from search_sliding import search, slide
import glob


def get_heatmap(image, bboxes):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)

    for b in bboxes:
        heatmap[b[0][1]:b[1][1], b[0][0]:b[1][0]] +=1
    return heatmap

def threshold_and_label(heatmap, threshold=THRESHOLD):
    heatmap[heatmap <= THRESHOLD] = 0
    labels = label(heatmap)
    return heatmap, labels


def get_bboxes(img, settings, num, clf, X_scaler, color_space=COLOR_SPACE, 
               spatial_size=SPATIAL_SIZE, hist_bins=HIST_BINS, 
               hist_range=BINS_RANGE, orient=ORIENT, 
               pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
               hog_channel=HOG_CHANNEL, spatial_feat=SPATIAL_FEAT, 
               hist_feat=HIST_FEAT, hog_feat=HOG_FEAT):
    res_img = np.copy(img)
    bboxes=[]

    for i in range(num):
        img_to_search = img ##cv2.resize(img[settings['y_limit'][i][0]:settings['y_limit'][i][1],:,:], settings['size'][i])
        windows = slide(res_img, settings['x_limit'][i], settings['y_limit'][i], settings['size'][i], settings['overlap'][i])
        hot = search(img_to_search, windows, clf, X_scaler, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        bboxes.extend(hot)
        res_img = draw_boxes(res_img, hot, color=(150,0,150), thickness=4)

    return bboxes, res_img

class Box():
    def __init__(self, first_box):
        self.final_box = [list(p) for p in first_box]
        self.count = 1
        self.boxes = [first_box]

    def get_count(self):
        return self.count

    def get_box(self):
        if len(self.boxes) > 1:
            center_box = np.average (np.average (self.boxes, axis=1), axis=0).astype(np.int32).tolist()
            xs = np.array(self.boxes) [:,:,0]
            ys = np.array(self.boxes) [:,:,1]

            half_x = int(np.std (xs))
            half_y = int(np.std (ys))

            box = ((center_box[0] - half_x, center_box[1] - half_y), (center_box[0] + half_x, center_box[1] + half_y))
            return box

    def should_join(self, box):
        x1 = box [0][0]
        y1 = box [0][1]
        x2 = box [1][0]
        y2 = box [1][1]
        xf1 = self.final_box [0][0]
        yf1 = self.final_box [0][1]
        xf2 = self.final_box [1][0]
        yf2 = self.final_box [1][1]
            
        x_overlap = max(0, min(xf2,x2) - max(xf1,x1))
        y_overlap = max(0, min(yf2,y2) - max(yf1,y1))
        
        area_cp = (x2 - x1) * (y2 - y1) * 0.3
        areaf_cp = (xf2 - xf1) * (yf2 - yf1) * 0.3
        intersection = x_overlap * y_overlap
        
        return (intersection >= area_cp or intersection >= areaf_cp)

    def join( self, boxes ):
        res = False

        for box in boxes:
            if self.should_join(box):
                boxes.remove(box)
                self.boxes.append(box)
                self.count += 1
                
                self.final_box[0][0] = min(self.final_box[0][0], box[0][0])
                self.final_box[0][1] = min(self.final_box[0][1], box[0][1])
                self.final_box[1][0] = min(self.final_box[1][0], box[1][0])
                self.final_box[1][1] = min(self.final_box[1][1], box[1][1])
                
                res = True

        return res

def get_final_boxes (bboxes, strength):
    final_boxes = []
    while len(bboxes) > 0:
        bbox = bboxes.pop (0)
        box = Box (bbox)
        while box.join (bboxes):
            pass
        final_boxes.append (box)
    
    boxes = []
    for box in final_boxes:
        if box.get_count () >= strength:
            boxes.append (box.get_box())
    return boxes

if __name__ == '__main__':
    with open('clf.pkl', 'rb') as fid:
        clf = pickle.load(fid)
    with open('scaler.pkl', 'rb') as fid:
        X_scaler = pickle.load(fid)

    images = glob.glob('test_images/*jpg')
    test_images=[]
    test_images_titles=[]

    for im in images:
        image = mpimg.imread(im)

        bboxes, bbox_img = get_bboxes(image, WINDOWS, NUM_WINDOWS, clf, X_scaler)
        test_images.append(bbox_img)

        heatmap = get_heatmap(bbox_img, bboxes)
        test_images.append(heatmap)

        final_boxes = get_final_boxes(bboxes, 2)
        final_image = draw_boxes( image, final_boxes, color=(255,0,80), thickness=4)
        test_images.append(final_image)
        test_images_titles.extend (['', '', ''])
    
    test_images_titles [0] = 'bounding boxes'
    test_images_titles [1] = 'heatmap'
    test_images_titles [2] = 'final boxes'        

    show_images_in_table (test_images, (3, 6), fig_size=(60, 72), titles=test_images_titles)

