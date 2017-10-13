**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/car.png
[car_yuv]: ./output_images/car_yuv.png
[car_hog_ch1]: ./output_images/car_hog_ch1.png
[car_hog_ch2]: ./output_images/car_hog_ch2.png
[car_hog_ch3]: ./output_images/car_hog_ch3.png
[noncar]: ./output_images/noncar.png
[noncar_yuv]: ./output_images/noncar_yuv.png
[noncar_hog_ch1]: ./output_images/noncar_hog_ch1.png
[noncar_hog_ch2]: ./output_images/noncar_hog_ch2.png
[noncar_hog_ch3]: ./output_images/noncar_hog_ch3.png
[detections1]: ./output_images/test1_detections.png
[detections2]: ./output_images/test2_detections.png
[detections3]: ./output_images/test3_detections.png
[detections4]: ./output_images/test4_detections.png
[detections5]: ./output_images/test5_detections.png
[detections6]: ./output_images/test6_detections.png

[heat_map1]: ./output_images/test1_heat_map_test.png
[heat_map2]: ./output_images/test2_heat_map_test.png
[heat_map3]: ./output_images/test3_heat_map_test.png
[heat_map4]: ./output_images/test4_heat_map_test.png
[heat_map5]: ./output_images/test5_heat_map_test.png
[heat_map6]: ./output_images/test6_heat_map_test.png

[slide1]: ./output_images/test1_sliding_window_test.png
[slide2]: ./output_images/test2_sliding_window_test.png
[slide3]: ./output_images/test3_sliding_window_test.png
[slide4]: ./output_images/test4_sliding_window_test.png
[slide5]: ./output_images/test5_sliding_window_test.png
[slide6]: ./output_images/test6_sliding_window_test.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `hog.py`

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![][car] ![][noncar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][car_yuv]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I trained using three scales of 1.0, 1.3 and 1.7. This can be seen in `sliding.py`.  

![alt text][slide3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I first converted the image to YCrCb color space, used only the first channel and extracted histogram, spatial and hog features. I use a heatmap to help with false detection

![][detections3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heat_map1]
![alt text][heat_map2]
![alt text][heat_map3]
![alt text][heat_map4]
![alt text][heat_map5]
![alt text][heat_map6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am not convinced HOG is the way to go on this one. I am thinking something more along the lines of DNN with backprop. This project was extremely frustrating, as my classifier would produce great scores, but not perform well on the moving images. It took a very long time to finish.

