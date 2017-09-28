# Camera Calibration with OpenCV
# ===
# 
# ### Run the code in the cell below to extract object points and image points for camera calibration.  

# In[1]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    print(idx,fname)
    img = cv2.imread(fname)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #write_name = 'out/corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()


# ### If the above cell ran sucessfully, you should now have `objpoints` and `imgpoints` needed for camera calibration.  Run the cell below to calibrate, calculate distortion coefficients, and test undistortion on an image!

# In[9]:


import pickle

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


#dst = cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imwrite('output_images/calibration_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)


# In[ ]:

#import pickle
#import cv2
#import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
#img = cv2.imread('test_image2.png')
#img = cv2.imread('camera_cal/calibration2.jpg')
img = cv2.imread('camera_cal/calibration13.jpg')
nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    ret, corners_orig = cv2.findChessboardCorners(img, (9,6),None)
    #print('--------')
    #print(corners_orig)
    dst = cv2.undistort(img, mtx, dist, None, None)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    #print('========')
    #print(corners)
    # 4) If corners found: 
    M = None
    warped = None
    if ret == True:
            # a) draw corners
            cv2.drawChessboardCorners(gray, (9, 6), corners, ret)
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            offset = 100 # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])

            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            #cv2.getPerspectiveTransform(src,dst,M)
	    print(src,dst)
            M = cv2.getPerspectiveTransform(src, dst)
	        # e) use cv2.warpPerspective() to warp your image to a top-down view
            #cv2.warpPerspective(gray, final, M)
            #print(nx,ny)
            warped = cv2.warpPerspective(gray, M,(1280,960))
    #delete the next two lines
    #M = None
    #img = warped
    #warped = np.copy(img) 
    return warped, M , gray

top_down, perspective_M, undistort_M= corners_unwarp(img, nx, ny, mtx, dist)
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=50)
#ax2.imshow(top_down)
#ax2.set_title('Undistorted and Warped Image', fontsize=50)
#ax2.imshow( undistort_M)
#ax2.set_title('Undistorted ', fontsize=50)
#plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

#print("after corner_unwarp")
#cv2.imshow('img', img)
#cv2.waitKey(500)
#cv2.imshow('top_down', top_down)
#cv2.waitKey(1000)
write_name = 'output_images/calibration13.jpg'
cv2.imwrite(write_name, top_down)



# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
 
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >mag_thresh[0]) & (gradmag < mag_thresh[1])] = 1

    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    #return color_binary
    return combined_binary

def warp(img,src,dst):
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M,(1280,720))
	return warped, M , Minv

# Process pipeline with aditional information
def process_image_ex(img,img2,img3):
    #undist = # function that returns undistorted image
    #img_binary, img_stack = # funcation that returns binary image (img_binary) and image with combination of all thresholds images (img_stack) - it will be displayed near process frame later
    #warped, Minv = # function that returns birds-eye view
    #lanes, ploty, left_fitx, right_fitx, left_curverad, right_curverad, center_dist = # function that detects lines and lane
    output = img# function that warp image back to perspective view

    output1 = cv2.resize(img2,(640, 360), interpolation = cv2.INTER_AREA)
    output2 = cv2.resize(img3,(640, 360), interpolation = cv2.INTER_AREA)

    # Create an array big enough to hold both images next to each other.
    vis = np.zeros([720, 1280+640, 3],dtype=np.uint8)
    #vis = np.zeros((720, 1280, 3))

    # Copy both images into the composed image.
    vis[:720, :1280,:] = output
    vis[:360, 1280:1920,:] = output1
    vis[360:720, 1280:1920,:] = output2
    return vis

def check_good_inds(leftx_current,rightx_current,margin,win_y_low,win_y_high,leftx_cr,rightx_cr):

	# right lane will refer to left cr, because it may be wrong now
	if leftx_cr > 0:
		xright_sign =1
	elif leftx_cr <0:
		xright_sign =-1
	else:
		xright_sign =0

	if rightx_cr > 0:
		xleft_sign =1
	elif rightx_cr <0:
		xleft_sign =-1
	else:
		xleft_sign =0

	win_xleft_low = leftx_current - margin + xleft_sign*margin/2
    	win_xleft_high = leftx_current + margin + xleft_sign*margin/2
    	win_xright_low = rightx_current - margin + xright_sign*margin/2
    	win_xright_high = rightx_current + margin + xright_sign*margin/2
	delta = 0
	if win_xright_low < win_xleft_high:
		delta =  win_xleft_high - win_xright_low
	win_xright_low += delta*2
	win_xright_high += delta*2
    	# Identify the nonzero pixels in x and y within the window
    	good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    	good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
	return win_xleft_low, win_xleft_high, win_xright_low,win_xright_high,good_left_inds,good_right_inds

 
# Read in an image and grayscale it
#image = mpimg.imread('test_images/straight_lines1.jpg')
#image = mpimg.imread('test_images/straight_lines2.jpg')
#image = mpimg.imread('test_images/test2.jpg')

#images = glob.glob('test_images/*.jpg')
#for img in images:
leftx_list=[]
rightx_list=[]
left_lane_inds = []
right_lane_inds = []
left_lane_inds_prev = []
right_lane_inds_prev = []
margin_range = 150#150
cap = cv2.VideoCapture('project_video.mp4')
# Define the codec and create VideoWriter object
#out = cv2.VideoWriter('project_video_output.avi', -1, 20.0, (1280,720))
do_process_image_ex = True
if do_process_image_ex == True:
	frame_width=1280 + 640
else:
	frame_width=1280
frame_height=720
leftx_list.append(margin_range*4)
rightx_list.append(frame_width - margin_range*4)
fourcc = cv2.cv.CV_FOURCC(*'MPEG')
out = cv2.VideoWriter('project_video_output.avi',fourcc, 30, (frame_width,frame_height))
count = 0
leftx_fit_cr=0
rightx_fit_cr=0
while cap.isOpened():
	print("before cap.read")
    	ret,frame = cap.read()
	if ret != 1:
		print("ret is not 1 for cap.read")
		#continue
		break
    	#cv2.imshow('window-name',frame)
    	#cv2.imwrite("frame%d.jpg" % count, frame)
    	count = count + 1
    	if cv2.waitKey(10) & 0xFF == ord('q'):
        	break
	#print(img)
	#image = mpimg.imread(img)
	#cvimage = cv2.imread(img)
    	image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    	image = cv2.undistort(image, mtx, dist, None, None)
	dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))   
	mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
	gradx_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
	grady_binary = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)
	hls_binary = hls_select(image, thresh=(90, 255))
	pipe_binary = pipeline(image)

	combined = np.zeros_like(dir_binary)
	combined[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

	#pre_warped_img = image
	pre_warped_img = pipe_binary
	img_size = (pre_warped_img.shape[1], pre_warped_img.shape[0])
	print(img_size)
	#src = np.float32([[255,687],[625,429],[654,429],[1041,677]])
	src = np.float32([[255,687],[609,444],[676,444],[1041,677]])
	#src = np.float32([[255,687],[586,460],[708,460],[1041,677]])
	#dst = np.float32([[255,687],[255,429],[1041,429],[1041,677]])
	newdstxl = (src[0][0]+src[1][0])/2
	newdstxr= (src[2][0]+src[3][0])/2
	newdstyd = pre_warped_img.shape[0]
	newdstyu= 0 #429#0
	dst = np.float32([[newdstxl,newdstyd],[newdstxl,newdstyu],[newdstxr,newdstyu],[newdstxr,newdstyd]])
	#warped_img = warp(pipe_binary,src,dst)
	warped_img, M, Minv = warp(pre_warped_img,src,dst)


	binary_warped = warped_img
	###  Find Lane

	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	print('leftx and rightx base : ',leftx_base, rightx_base)

	# Choose the number of sliding windows
	nwindows = 9
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	if len(leftx_list) > 9:
		if abs(leftx_current - leftx_list[-9]) > minpix:
			leftx_current = leftx_list[-9]
	if len(rightx_list) > 9:
		if abs(rightx_current - rightx_list[-9]) > minpix:
			rightx_current = rightx_list[-9]
	leftx_list.append(leftx_current)
	rightx_list.append(rightx_current)
 	leftx_avg=np.int(np.mean(leftx_list))	
 	rightx_avg=np.int(np.mean(rightx_list))	
	print(" -- leftx leftx_avg rightx rightx_avg : ",leftx_current,leftx_avg,rightx_current,rightx_avg)
	if (leftx_avg - leftx_current ) > margin_range:
		leftx_current = leftx_avg
	if (rightx_current - rightx_avg) > margin_range:
		rightx_current = rightx_avg
	print(" --- leftx leftx_avg rightx rightx_avg : ",leftx_current,leftx_avg,rightx_current,rightx_avg)
	# Create empty lists to receive left and right lane pixel indices
	import copy
	if left_lane_inds != []:
		print(" len left land inds , len prev inds",len(left_lane_inds),len(left_lane_inds_prev))
		left_lane_inds_prev = copy.copy(left_lane_inds)
	if right_lane_inds != []:
		print(" len right land inds , len prev inds",len(right_lane_inds),len(right_lane_inds_prev))
		right_lane_inds_prev = copy.copy(right_lane_inds)
	left_lane_inds = []
	right_lane_inds = []

	invalid_left_windows = []
	invalid_right_windows = []
	# Step through the windows one by one
	err_msg=""
	for window in range(nwindows):
    		# Identify window boundaries in x and y (and right and left)
    		win_y_low = binary_warped.shape[0] - (window+1)*window_height
    		win_y_high = binary_warped.shape[0] - window*window_height


		win_xleft_low, win_xleft_high, win_xright_low,win_xright_high,good_left_inds,good_right_inds = check_good_inds(leftx_current,rightx_current,margin,win_y_low,win_y_high,0,0)
    		# Append these indices to the lists
		#print("######### window minpix ,len of good_left_inds and good_right_inds",window,minpix,len(good_left_inds),len(good_right_inds))
    		# If you found > minpix pixels, recenter next window on their mean position
    		if len(good_left_inds) > minpix:
        		leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		else:
			win_xleft_low, win_xleft_high, win_xright_low,win_xright_high,good_left_inds,good_right_inds = check_good_inds(leftx_list[-1],rightx_list[-1],margin,win_y_low,win_y_high,0,rightx_fit_cr)
    			if len(good_left_inds) > minpix:
        			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			else:
				invalid_left_windows.append(window)
			
    		if len(good_right_inds) > minpix:        
        		rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		else:
			win_xleft_low, win_xleft_high, win_xright_low,win_xright_high,good_left_inds,good_right_inds = check_good_inds(leftx_list[-1],rightx_list[-1],margin,win_y_low,win_y_high,leftx_fit_cr,0)
    			if len(good_right_inds) > minpix:        
        			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
			else:
				invalid_right_windows.append(window)

		print("######### window minpix ,len of good_left_inds and good_right_inds",window,minpix,len(good_left_inds),len(good_right_inds))
    		# Draw the windows on the visualization image
    		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    		left_lane_inds.append(good_left_inds)
    		right_lane_inds.append(good_right_inds)
    		#print("leftx and rightx current: ",leftx_current,rightx_current)
		#print(" == window ,leftx leftx_avg rightx rightx_avg : ",window,leftx_current,leftx_avg,rightx_current,rightx_avg)
		#if (leftx_avg - leftx_current ) > margin_range:
		#	leftx_current = leftx_avg
		#if (rightx_current - rightx_avg) > margin_range:
		#	rightx_current = rightx_avg
		#print(" ===== window ,leftx leftx_avg rightx rightx_avg : ",window,leftx_current,leftx_avg,rightx_current,rightx_avg)
		leftx_list.append(leftx_current)
		rightx_list.append(rightx_current)
		if window == 1:
			bottom_leftx = leftx_current
			bottom_rightx = rightx_current
			
	nwindows_img = out_img
	print(" size of lefx, size of rightx , bottom leftx rightx .",len(leftx_list),len(rightx_list),bottom_leftx,bottom_rightx)
	# Concatenate the arrays of indices
	print(" before concatenate left lane inds len , right lane inds len",len(left_lane_inds),len(right_lane_inds))
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)
	# Extract left and right line pixel positions
	print(" left lane inds len , right lane inds len",len(left_lane_inds),len(right_lane_inds))
	print(" left lane inds last , right lane inds last",left_lane_inds[-1],right_lane_inds[-1])
	print(" nonzerox len, nonzeroy len",len(nonzerox),len(nonzeroy))
	rcount =0
	max_rcount = 0 
	#new_right_lane_inds=[]
	#new_left_lane_inds=[]
	for i in right_lane_inds:
		if i > len(nonzerox):
			#new_right_lane_inds = np.delete(right_lane_inds ,rcount)
			print(rcount,i)
			if rcount > max_rcount:
				max_rcount = rcount
		rcount+=1
	#print(" right lane rcount",rcount)
	lcount =0
	max_lcount = 0 
	for i in left_lane_inds:
		if i > len(nonzerox):
			#new_left_lane_inds = np.delete(left_lane_inds ,rcount)
			print(lcount,i)
			if lcount > max_lcount:
				max_lcount = lcount
		lcount+=1
	
	#print(" left lane rcount",lcount)
		#print i
	left_lane_inds = left_lane_inds[max_lcount+1:]
	right_lane_inds = right_lane_inds[max_rcount+1:]
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	#print("left line : ", len(leftx),len(lefty),leftx,lefty)
	# Fit a second order polynomial to each
	if len(invalid_left_windows) == 0:#nwindows/3:
		err_msg += "Left Lane invalid windows number : "+str(len(invalid_left_windows))
		left_fit = np.polyfit(lefty, leftx, 2)
	if len(invalid_right_windows) == 0:#nwindows/3:
		err_msg += "Right Lane invalid windows number : "+str(len(invalid_right_windows))
		right_fit = np.polyfit(righty, rightx, 2)
	print(err_msg)
	#left_fit = np.polyfit(lefty, leftx, 2)
	#right_fit = np.polyfit(righty, rightx, 2)
	print("\n np polyfit \n")
	print(left_fit)
	print(right_fit)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	print("\n  left_fitx and right_fity \n")
	#print(left_fitx)
	#print(right_fitx)

	#### find curve

	y_eval = np.max(ploty)
	print(y_eval)
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = (50.0/720) # meters per pixel in y dimension
	xm_per_pix = (3.7/400) # meters per pixel in x dimension
	#print(ym_per_pix,xm_per_pix)
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	cr_msg = "left fit cr : " + str(left_fit_cr[0]) + "right fit cr : "+str(right_fit_cr[0])
	leftx_fit_cr=left_fit_cr[0]
	rightx_fit_cr=right_fit_cr[0]
	#left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / 2*left_fit_cr[0]
	#right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / 2*right_fit_cr[0]
	# Now our radius of curvature is in meters
	print(left_curverad, 'm', right_curverad, 'm')
	# Example values: 632.1 m    626.2 m
	
	#### find center
	bottom_y = 720
	bottom_leftx = left_fit[0]*bottom_y**2 + left_fit[1]*bottom_y + left_fit[2]
	bottom_rightx = right_fit[0]*bottom_y**2 + right_fit[1]*bottom_y + right_fit[2]
	#print(" === bottom leftx rightx  ",bottom_leftx,bottom_rightx )
	bottom_center = ( bottom_leftx + bottom_rightx )/2
	bottom_midpoint = np.int(histogram.shape[0]/2)
	print(" - bottom_midpoint",bottom_midpoint)
	right_of_bottom_midpoint_meter = (bottom_center - bottom_midpoint)*xm_per_pix

	show_curve_line = False
	if show_curve_line == True:
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
		plt.imshow(out_img)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()




	# Next frame Skip sliding windows search
	# Assume you now have a new warped binary image 
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!


	# Vistualize it
	# Create an image to draw on and an image to show the selection window
	#show_curve_line_noslide = False
	#if show_curve_line_noslide == True:
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	fitx_img = result

	show_curve_line_noslide = False
	if show_curve_line_noslide == True:
		plt.imshow(result)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.show()


	##### Draw final image
	warped = binary_warped

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	#result = color_warp
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	#result=newwarp
	# Combine the result with the original image
	result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
	plt.imshow(result)
	#plt.show()
    	displayimage = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
	# Write some Text

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,50)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2
	msg = "left curvature : " + str(left_curverad) +" , " + "right curvature : " + str(right_curverad)

	cv2.putText(displayimage,msg, 
    	bottomLeftCornerOfText, 
    	font, 
    	fontScale,
    	fontColor,
    	lineType)
	msg2 = " vehicle is "+str(right_of_bottom_midpoint_meter) +"m right of center"+err_msg
	bottomLeftCornerOfText = (10,100)
	cv2.putText(displayimage,msg2, 
    	bottomLeftCornerOfText, 
    	font, 
    	fontScale,
    	fontColor,
    	lineType)


	if do_process_image_ex == True:
		displayimage = process_image_ex(displayimage,nwindows_img,fitx_img)
    	#cv2.imshow('window-name',displayimage)
    	cv2.imshow('window-name',displayimage)
    	cv2.waitKey(50)
	# write the display frame
        out.write(displayimage)
cap.release()
out.release()
cap.destroyAllWindows()
