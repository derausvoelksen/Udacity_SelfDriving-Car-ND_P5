##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./sliding_window_search.PNG
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/hog_viz.PNG
[image9]: ./output_images/sliding_window_64.png
[image10]: ./output_images/sliding_window_96.png
[image11]: ./output_images/sliding_window_128.png
[image12]: ./output_images/sliding_window_196.png
[image13]: ./output_images/pipeline_single_frame.PNG
[image14]: ./output_images/heatmap_1.PNG
[image15]: ./output_images/heatmap_2.PNG
[image16]: ./output_images/heatmap_3.PNG
[image17]: ./output_images/heatmap_4.PNG
[image18]: ./output_images/heatmap_5.PNG


[video1]: ./project_video_out_cs.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In cell 3 all the `vehicle` and `non-vehicle` images are read in.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The code for the HOG feature extract is contained in the fifth code cell of the IPython notebook VehicleDetect.ipynb.



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I got the advice, that RGB color space wasnt the best choice, so I turned to try out YUV. I tried to benchmark with the parameters `orient`, `pix_per_cell`, `cell_per_block`, and found out, that having set the below quoted parameters are the fastest on my machine, but still resulted in a decent vehicle detection. Especially the histogram and spatial binning did not enhance the detection but caused a larger feature vector:


```python
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = False # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
```

Cell four shows an example of a HOG Vizualization:
![HOG Vizualization][image8]


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in cell 8 and achieved a accuracy of 98,59% on the test data set. I did not use spacial binning and histogram features.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I printed several window sizes and regions in order to determine size and area within the camera image. Dependent on the distance to the ego vehicle, I chose only a small region, so the computation would be as fast as possible. Here are the resulting areas within the image:

* Windows of size 64x64:
![Sliding Window (64x64)][image9]
* Windows of size 96x96:
![Sliding Window (96x96)][image10]
* Windows of size 128x128:
![Sliding Window (128x128)][image11]
* Windows of size 196x196:
![Sliding Window (196x196)][image12]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I created the `pipeline_single_frame` function in order to automate the steps of feature extraction, applying the heatmap, labeling and drawing the bounding boxes onto the image (cell 25).

![Pipeline][image13]

As stated above, I did not make use of the histogram and spatial binning, and chose carefully the sliding window area within the image. The HOG parameters were chosen so that the detection of vehicles still work, but also result in a decent computation time (but not real time on my computer).


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_cs.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For eliminating false positive detections, I first used the heatmap and labeling.

### Here is a sequence of images showing the heatmap in operation:

![Feature extraction][image14]
![applying heatmap][image15]
![applying threshold][image16]
![applying lables][image17]
![drawing bounding boxes][image18]

Depending on the number of independent labels (output of `labels[1]`), there are drawn boxes around the most outer pixels if the label area. If a heat spot is not separated by black pixels, then it will be detected as a single car.


In order to further avoid false positives the detections of the past 15 frames are stored and added to a heatmap (see function `pipeline_history`. Then, a threshold is applyed to this heatmap to cut out wrong detections over the past frames.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It was hard to find a decent balance between computation time and detection accuracy, and I believe it still could be improved. One idea would be to have a dynamic search around a detected vehicle taking its estimated velocity into account.

The pipeline will probably fail on situations that have not been trained (rain, night, sunlight,..).

