**Vehicle Detection Project**
I reused a lot of the code from the lessons, as I already coded them and it was my own work.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[heat]: ./examples/heat.png
[hog]: ./examples/hog.png
[search_sliding]: ./examples/search_sliding.png
[train] ./examples/train.png

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I want to start with the discussion.  This project took me many hours to complete...and I actually completed it twice. First, I followed the guidelines from the lessons.  That produced high test accuracy from my classifier, but I couldn't get the boxes to be accurate on moving images no matter what.
After doing a bunch of research and discussing with other classmates, I decided that the main problems were caused by my feature sets. Color saturation was messing with it, so I used grayscale for HOG extraction since this would retain the structural information. I also reduced my orient size to 8 and cell per block to only 1. I didn't use spatial bins or color histograms at all.  My predictions processed much faster and I was able to accurately draw boxes on moving images


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `features.py`

I started by reading in all the `vehicle` and `non-vehicle` images.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `GRAY` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`:

![alt text][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally decided to use grayscale and minimal hog parameters for slight influence.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using hog features

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I trained using three window scales: 128x128, 96x96 and 80x80

![alt text][search_sliding]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I first converted the image to GREY color space and hog features. I applied a StandardScaler to the feature values, though probably not necessary since I didn't use spatial bins or histograms. I use a heatmap to help with false detection and merged overlapping boxes

![][heat]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also joined nearby bounding boxes to create on averaged box.

I also kept a queue of 10 most current frames, using overlapping boxes to determine true positives.



---

