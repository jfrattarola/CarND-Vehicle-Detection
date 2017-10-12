from parameters import *
import matplotlib.image as mpimg
import numpy as np
from features import convert_color, get_hog_features

if __name__ == '__main__':
    car_img = mpimg.imread('vehicles/MiddleClose/image0000.png')
    cvt_car_img = convert_color(car_img, COLOR_SPACE)
    non_car_img = mpimg.imread('non-vehicles/MiddleClose/image0000.png')
    cvt_noncar_img = convert_color(non_car_img, COLOR_SPACE)

    ch1_car = car_img[:, :, 0]
    ch2_car = car_img[:, :, 1]
    ch3_car = car_img[:, :, 2]
    car_features1, car_hog_image1 = get_hog_features(ch1_car, vis=True)
    car_features2, car_hog_image2 = get_hog_features(ch2_car, vis=True)
    car_features3, car_hog_image3 = get_hog_features(ch3_car, vis=True)

    ch1_noncar = cvt_noncar_img[:, :, 0]
    ch2_noncar = cvt_noncar_img[:, :, 1]
    ch3_noncar = cvt_noncar_img[:, :, 2]
    noncar_features1, non_car_hog_image1 = get_hog_features(ch1_noncar, vis=True)
    noncar_features2, non_car_hog_image2 = get_hog_features(ch2_noncar, vis=True)
    noncar_features3, non_car_hog_image3 = get_hog_features(ch3_noncar, vis=True)

    print('car_features1 {}'.format(car_features1.shape))
    print('car_features2 {}'.format(car_features2.shape))
    print('car_features3 {}'.format(car_features3.shape))

    mpimg.imsave('output_images/car_hog_ch1.png', car_hog_image1, cmap='gray')
    mpimg.imsave('output_images/car_hog_ch2.png', car_hog_image2, cmap='gray')
    mpimg.imsave('output_images/car_hog_ch3.png', car_hog_image3, cmap='gray')
    mpimg.imsave('output_images/car.png', cvt_car_img)

    mpimg.imsave('output_images/noncar_hog_ch1.png', non_car_hog_image1, cmap='gray')
    mpimg.imsave('output_images/noncar_hog_ch2.png', non_car_hog_image2, cmap='gray')
    mpimg.imsave('output_images/noncar_hog_ch3.png', non_car_hog_image3, cmap='gray')
    mpimg.imsave('output_images/noncar.png', cvt_noncar_img)
