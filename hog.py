from parameters import *
from utils import *
import matplotlib.image as mpimg
import numpy as np
from features import get_hog_features, get_color_channel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()
    
    
    examples = list(get_examples(FLAGS.dir)[1:-1])
    feature_examples = []
    feature_examples.extend(examples)

    for e in examples:
        ch = get_color_channel( e, 'HLS', 2 )
        features, hog_image = get_hog_features(ch, ORIENT, PIX_PER_CELL, CELL_PER_BLOCK, vis=True)
        feature_examples.append(hog_image)

    show_images_in_table( feature_examples, (6,2), fig_size=(20,6) )
