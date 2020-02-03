# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pyrealsense2 as rs
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)  
    plt.show()



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

file_prefix = 'image_'
video_file = 'video_1'
coordinates = {} 
out = cv2.VideoWriter(video_file,
                      cv2.VideoWriter_fourcc('M','J','P','G'), 
                      10, (640,480))

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter 
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

depth_list = np.zeros(shape=(480,640))

idx = 0 
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
#        depth_frame = frames.get_depth_frame()
#        color_frame = frames.get_color_frame()
#        if not depth_frame or not color_frame:
#            continue
#
#        # Convert images to numpy arrays
#        depth_image = np.asanyarray(depth_frame.get_data())
#        color_image = np.asanyarray(color_frame.get_data())
#

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        depth_list = np.dstack([depth_list,depth_image])

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        
        # Save image, depth and color images
        image_file = file_prefix + str(idx) + '.png'
        depth_file = file_prefix + str(idx) + '_depth' + '.png'
        color_file = file_prefix + str(idx) + '_color' + '.png'
#        cv2.imwrite(image_file, images)
#        cv2.imwrite(depth_file, depth_image)
#        cv2.imwrite(color_file, color_image)
#        
        # Save frame to video file
#        out.write(depth_frame)
        
        # Show images for debugging
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        idx += 1 
finally:

    # Stop streaming
    pipeline.stop()



#test = depth_list[:,:,30]    
#plt.imshow(test)
#
#test_color = cv2.applyColorMap(cv2.convertScaleAbs(test, alpha=0.03), cv2.COLORMAP_JET)
#plt.imshow(test_color)
#
#test_bg = test.copy()
#test_bg[np.where(test_bg > 420)] =0
#plt.imshow(test_bg)

df =np.load('sample_stream.npy')
new_df = []

ndimage.measurements.center_of_mass(test_bg)

count = 0 
for frame in range(df.shape[2]):
    test = df[:,:,frame]
    test_bg = test.copy()
    test_bg[np.where(test_bg > 450)] = 0
    new_df.append(test_bg)
    filename = 'image_' + str(count) + '.png'
    cv2.imwrite(filename, test_bg)
    count += 1
    

def calculateCoM(self, dpt):
    """
    Calculate the center of mass
    :param dpt: depth image
    :return: (x,y,z) center of mass
    """

    dc = dpt.copy()
    dc[dc < self.minDepth] = 0
    dc[dc > self.maxDepth] = 0
    cc = ndimage.measurements.center_of_mass(dc > 0)
    num = numpy.count_nonzero(dc)
    com = numpy.array((cc[1]*num, cc[0]*num, dc.sum()), numpy.float)

    if num == 0:
        return numpy.array((0, 0, 0), numpy.float)
    else:
        return com/num
