## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./examples/car_notcar.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/single_scale_detection.png
[image4]: ./examples/multi_scale_detection.png
[image5]: ./examples/heatmap.png
[image6]: ./examples/final_detection.png
[video1]: ./projectoutput_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The functions for extracting the HOG features step are contained in section [HOG features](Submission%20code.ipynb#HOG-features). 

I started by reading in all the `vehicle` and `non-vehicle` images as described in the section [Dataset](Submission%20code.ipynb#Dataset).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. This is described in the section [Feature extraction](Submission%20code.ipynb#Feature-extraction) . To store the various parameters and pass them around conveniently, I have created the `Parameters` class which encapsulates all the required parameters for the project.

Finally, [here](Submission%20code.ipynb#Extract-and-visualise) is an example using the `Y` channel of `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, both for the color space as well HOG parameters. I started by using a concatenated feature of HOG, spatial binning and color histogram.  

For color spaces, the best results were obtained with `YCrCb` and `YUV` and so further experiments were carried out only on these.

The combination of HOG, spatial features and color histogram features provided good accuracy on the test set, however they produced a lot of false positives when used on real images.

In the end, I ditched spatial features and color histogram features and only used the HOG features. For HOG, when the number of orientations and pixels per cell were low, the accuracy was good, but the speed of training and prediction was very slow. In the end, I chose `orientations` to be 11, `pixels_per_cell` to be 16. `cells_per_block` with a value of 2 was satisfactory. These set of parameters provided a good balance between accuracy and speed.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the SVM using the HOG features is in section [Classifier](Submission%20code.ipynb#Classifier).

I first extract and stack the HOG features from all the vehicle and non-vehicle images in dataset. This is then (shuffled and) split into train and test sets, with a test set comprising 20% of the dataset.

I trained a linear SVM using `sklearn.svm.LinearSVC` and the HOG features extracted above using the default parameters. The accuracy on the test set was 98.14%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window search is in function `find_cars` in this [section](Submission%20code.ipynb#Function-to-find-cars) 

Instead of extracting HOG features of the windows individually, it extracts the HOG of the entire ROI and then sub-samples the features for each window. This speeds up the process considerably. It then predicts the presence of car using the classifier trained above for each of the window positions. The positions of each of the positive window predictions are recorded and returned. Here is an example of detection on a single scale

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I then tried out detection at multiple scales. Using scales of less than 1 provided sub-optimal results with too many false positives. The ROI also affected the outcome of the detection. 

Ultimately I decided to adopt the strategy by Jeremy Shannon as described [here](https://github.com/jeremy-shannon/CarND-Vehicle-Detection). This searches at scales of `[1.0, 1.5, 2.0, 3.5]` with two different ROIs for each scale. I used with 3-channel HOG features of `YUV` images.

Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./projectoutput_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for the video pipeline is in section [Video pipeline](Submission%20code.ipynb#Video-pipeline). 

I have created a class `VehicleDetector` that stores the parameters and previous frame history, and performs frame by frame detection in the method `processFrame`. This results in a list of bounding boxes of positive detections for each image. These bounding boxes are also stored in a buffer of previously detected bounding boxes, with a buffer capacity of 15 frames. 

The bounding boxes buffer is required to minimize the effect of false positives on the final result. The idea behind this is that false positives lead to sparse and sporadic boxes, while true positives lead to dense and consistent boxes. We add up all the previous detections from the buffer with the current detection to generate a "heatmap", which identifies regions of dense detections. This technique is also called Non-Maximum Suppression. An example of a heatmap is shown below:

![heat map][image5]



I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  This leads to the final detections as shown below:

![final detection][image6]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach is pretty much the same as suggested in the lectures. However, I decided not to use spatial binned and color histogram features as they led to a high rate of false positives. I also had a trouble with the classifier when the HOG features I extracted were extracted from `scipy`'s implementation using the parameter `block_norm == L2-Hys`. This led to a significant drop in classifier accuracy for some reason. 

Although the pipeline should be quite robust to lighting and color changes, there might be some corner cases where this may fail. It did have trouble finding the white car with certain parameter combination in some of the experiments. This mainly depends on the classifier accuracy. Therefore, a better classifier will most likely lead to better detections. Also, the data on which the classifier is trained is also important.

Finally, the video pipeline is not real-time. To make it real-time, some heavy optimization is required, both in the methodology used, as well the implementation.

In the  future versions of this project, I plan to use deep learning methods for better and faster detections. I'm interested in looking at the performance of [YOLO2](https://arxiv.org/abs/1612.08242). YOLO has been known to provide extremely fast and accurate detections. This could be a method that might make the video pipeline run in real-time.

