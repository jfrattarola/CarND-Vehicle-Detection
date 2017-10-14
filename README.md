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

I tried various combinations of parameters and finally decided to use grayscale and minimal hog parameters for slight influence. Here is a list of all the parameters/configurations I used:
```
COLOR_SPACE='GRAY'
HOG_CHANNEL='ALL'
ORIENT=8
PIX_PER_CELL=16
CELL_PER_BLOCK=1
SPATIAL_SIZE=(16, 16)
HIST_BINS=16
BINS_RANGE=(0,1)
YSTART = 400
YSTOP = 650
THRESHOLD=5
DEFAULT_BOX_COLOR=(0,0,255)
DEFAULT_BOX_THICKNESS=6
SPATIAL_FEAT=False
HIST_FEAT=False
HOG_FEAT=True
WINDOWS={}
WINDOWS['x_limit']= [[None, None], [32, None], [412, 1280]]
WINDOWS['y_limit'] = [[400,640], [400,600], [390,540]]
WINDOWS['size'] = [(128,128), (96,96), (80,80)]
WINDOWS['overlap'] = [(0.5,0.5), (0.5,0.5), (0.5,0.5)]
NUM_FRAMES=10
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the default rbf and kernel paramters from sklearn. This code is in `train.py` and `test.py`. It uses `data.py` to gather training data. I used 25% for testing. I also used the StandardScaler.  My classifier got 99.1% accuracy

```
python train.py --dir=/vol/data
Loading files from /vol/data/
extracting features for 8792 cars
extracting features for 8968 non-cars

Number of training samples: 13320
Number of test samples: 4440
Number of positive samples: 8792
Training took 4.837543725967407 seconds and produced an accuracy of 0.991
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I trained using three window scales: 128x128, 96x96 and 80x80 with 50% overlapping while sliding. I also only look at the x/y coordinates where cars should be.

In this image, you can see the search area for each window area, and the size of a sliding window.

![alt text][search_sliding]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I first converted the image to GREY color space and hog features.  I use a heatmap to help with false detection and merged overlapping boxes

![][heat]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./final_video.m4v)

[![Vehicle detection and tracking](http://img.youtube.com/vi/xBobUUFdofo/0.jpg)](https://www.youtube.com/watch?v=xBobUUFdofo)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I also joined nearby bounding boxes to create on averaged box.

I also kept a queue of 10 most current frames, using overlapping boxes to determine true positives.



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I want to start with the discussion.  This project took me many hours to complete...and I actually completed it twice. First, I followed the guidelines from the lessons.  That produced high test accuracy from my classifier, but I couldn't get the boxes to be accurate on moving images no matter what.
After doing a bunch of research and discussing with other classmates, I decided that the main problems were caused by my feature sets. Color saturation was messing with it, so I used grayscale for HOG extraction since this would retain the structural information. I also reduced my orient size to 8 and cell per block to only 1. I didn't use spatial bins or color histograms at all.  My predictions processed much faster and I was able to accurately draw boxes on moving images




