
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image0]: ./camera_cal/calibration13.jpg "Undistorted"
[image1]: ./output_images/calibration13.jpg "calibrated"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image2_1]: ./output_images/undist_test2.jpg "Road Transformed"
[image3]: ./output_images/binary_test2.jpg "Binary Example"
[image4]: ./output_images/warped_test2.jpg "Warp Example"
[image5]: ./output_images/video_snapshot.png  "Fit Visual"
[image6]: ./output_images/video_snapshot_v2.png "v2 Output"
[image7]: ./output_images/video_snapshot_v4.png "v4 Output"
[video1]: ./project_video_output_v4.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained  in lines #13 through #174 of the file called `adv_lane_finding.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
below is an test image:

![alt text][image0]

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
here is the image after undistorted, and the deer sign has been undistorted and unwarped.
![alt text][image2_1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #178 through #274 in `another_file.py`), and the final pipeline() function is at line #248  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines #276 through #280 in the file `adv_lane_finding.py` .  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[255,687],[609,444],[676,444],[1041,677]])

newdstxl = (src[0][0]+src[1][0])/2
newdstxr= (src[2][0]+src[3][0])/2
newdstyd = pre_warped_img.shape[0]
newdstyu= 0 
dst = np.float32([[newdstxl,newdstyd],[newdstxl,newdstyu],[newdstxr,newdstyu],[newdstxr,newdstyd]])

```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

test image :

![alt text][image3]

warped image : 

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did windows sliding  and historgram to fit my lane lines with a 2nd order polynomial kinda like this at line #432 through line #687:

![alt text][image5]

I found that my detected lane lines may be wrong and detected to another lane line or the edge of the highway.


![alt text][image6]

First, we only caculate the new curve if there is no invalid sliding window result which mean enough good indidation points
```python
	if len(invalid_left_windows) == 0:
		left_fit = np.polyfit(lefty, leftx, 2)
	
	if len(invalid_right_windows) == 0:
		right_fit = np.polyfit(righty, rightx, 2)
                
```

Second , we keep right and left search windows within a average range, and will adjust right or left window position according the previous searching windows condition.
```python
	current_window_distance = win_xright_low - win_xleft_high
	win_dist_diff = current_window_distance - bottom_window_distance
	if abs(win_dist_diff) > margin:
		if invalid_left_windows_number > 0:
			win_xleft_low += win_dist_diff
			win_xleft_high += win_dist_diff
		elif invalid_right_windows_number > 0:
			win_xright_low += win_dist_diff
			win_xright_high += win_dist_diff  	
		else:
			win_xright_low += win_dist_diff/2
			win_xright_high += win_dist_diff/2 	
			win_xleft_low += win_dist_diff/2
			win_xleft_high += win_dist_diff/2
			print(" !!!!! Issue of right or left window position  delta:",current_window_distance - 			
                
```


Moreover, I gave an initial average for both left and right lane position.
```python
frame_width=1280
frame_height=720
leftx_list.append(margin_range*4)
rightx_list.append(frame_width - margin_range*4)

```
here is the final result.

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #603 through #622 in my code in `adv_lane_finding.py` for cacaluating the curvature
I did this in lines #624 through #632 in my code in `adv_lane_finding.py` for caculating the position of the vehcile with respect of the center

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
the video is too large for Github, so I didn't upload it to github

Here's a [link to my video result](./project_video_output_v4.avi)

![alt text][video0]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a problems about detecting wrong lane lines while there is another lane line with brighter color or a bright road edge.
I prevent this wrong detection by keeping left and right searching windows within a range and also use previous curve line if there is any none valid search windows result.

I also gave a initial average lane position, so the first frame will also have correct detection.
