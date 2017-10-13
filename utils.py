import random
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

def show_images_in_table (images, table_size, fig_size = (10, 10), cmap=None, titles=None):
    sizex = table_size [0]
    sizey = table_size [1]
    fig, imtable = plt.subplots (sizey, sizex, figsize = fig_size, squeeze=False)
    for j in range (sizey):
        for i in range (sizex):
            im_idx = i + j*sizex
            if (isinstance(cmap, (list, tuple))):
                imtable [j][i].imshow (images[im_idx], cmap=cmap[i])
            else:
                im = images[im_idx]
                if len(im.shape) == 3:
                    imtable [j][i].imshow (im)
                else:
                    imtable [j][i].imshow (im, cmap='gray')
            imtable [j][i].axis('off')
            if not titles is None:
                imtable [j][i].set_title (titles [im_idx], fontsize=32)

    plt.show ()

def plt_show_gray (image):
    plt.figure ()
    plt.imshow (image, cmap='gray')
    plt.show ()

def plt_show (image):
    plt.figure ()
    plt.imshow (image)
    plt.show ()


def get_examples(path):
    cars = glob.glob('{}/vehicles/**/*.png'.format(path), recursive=True)
    notcars = glob.glob('{}/non-vehicles/**/*.png'.format(path), recursive=True)
    
    # cloading car images
    car_image = []
    for impath in cars:
        car_image.append (mpimg.imread(impath))

    # loading non car images
    notcar_image = []
    for impath in notcars:
        notcar_image.append (mpimg.imread(impath))

    car_image_count = len (car_image)
    notcar_image_count = len (notcar_image)
    
    print ('dataset has cars:', car_image_count)
    print ('none cars:', notcar_image_count)

    # show dataset images examples
    example_images = [
        car_image [random.randint (0, car_image_count-1)],
        car_image [random.randint (0, car_image_count-1)],
        car_image [random.randint (0, car_image_count-1)],
        car_image [random.randint (0, car_image_count-1)],
        
        notcar_image [random.randint (0, notcar_image_count-1)],
        notcar_image [random.randint (0, notcar_image_count-1)],
        notcar_image [random.randint (0, notcar_image_count-1)],
        notcar_image [random.randint (0, notcar_image_count-1)]
        ]
    return example_images
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to read vehicle/non-vehicle image files from')
    FLAGS, unparsed = parser.parse_known_args()
    
    
    examples = get_examples(FLAGS.dir)
    show_images_in_table (example_images, (4, 2), fig_size=(20, 10))
