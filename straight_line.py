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



# Read in an image and grayscale it
#image = mpimg.imread('test_images/straight_lines1.jpg')
#image = mpimg.imread('test_images/straight_lines2.jpg')
image = mpimg.imread('test_images/test2.jpg')

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
	warped = cv2.warpPerspective(img, M,(1280,720))
	return warped
 

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
warped_img = warp(pre_warped_img,src,dst)

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax1.plot(src[0][0],src[0][1],'+')
ax1.plot(src[1][0],src[1][1],'+')
ax1.plot(src[2][0],src[2][1],'+')
ax1.plot(src[3][0],src[3][1],'+')

#ax1.imshow(pipe_binary, cmap='gray')
#ax1.set_title('image before warp', fontsize=50)
#ax2.imshow(grad_binary, cmap='gray')
#ax2.set_title('Thresholded Gradient', fontsize=50)
#ax2.imshow(mag_binary, cmap='gray')
#ax2.set_title('ThresholdedMap', fontsize=50)
#ax2.imshow(dir_binary, cmap='gray')
#ax2.set_title('Thresholded Dir', fontsize=50)
#ax2.imshow(combined, cmap='gray')
#ax2.set_title('Thresholded combined', fontsize=50)
#ax2.imshow(hls_binary, cmap='gray')
#ax2.set_title('Thresholded HLS', fontsize=50)
#ax2.imshow(pipe_binary, cmap='gray')
#ax2.set_title('Thresholded pipe', fontsize=50)
#ax2.imshow(warped_img, cmap='gray')
ax2.imshow(warped_img)
ax2.set_title('warped pipe', fontsize=50)
ax2.plot(dst[0][0],dst[0][1],'+')
ax2.plot(dst[1][0],dst[1][1],'+')
ax2.plot(dst[2][0],dst[2][1],'+')
ax2.plot(dst[3][0],dst[3][1],'+')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
