#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:59:57 2020

@author: boonyew
"""

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
#sys.path.append('/usr/local/python')
from openpose import pyopenpose as op

import pyrealsense2 as rs
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../../models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

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

try:
    while True:
        start_time = time.time()
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
        
        imageToProcess = color_image
        handRectangles = [
            # Left/Right hands person 0
            [
            op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
            op.Rectangle(0., 0., 0., 0.),
            ],
            # Left/Right hands person 1
            [
            op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
            op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
            ],
            # Left/Right hands person 2
            [
            op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
            op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
            ]
        ]
        
        # Create new datum
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.handRectangles = handRectangles
        
        # Process and display image
        opWrapper.emplaceAndPop([datum])
        print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
        print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)        
        print("FPS: ", 1.0 / (time.time() - start_time))
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        
        # Show images for debugging
#        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        idx += 1 
finally:

    # Stop streaming
    pipeline.stop()