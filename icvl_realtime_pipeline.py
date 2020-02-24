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

#import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
h5_path =  './result/icvl_clstm_basic_120.hdf5'
model = load_model(h5_path)
model.summary()
#icvl_net = cv2.dnn.readNetFromTensorflow('./new/model.pb')

height = 480
width= 640
cropHeight = 176
cropWidth = 176
fx=628.668
fy=628.668
u0=311.662
v0=231.571
xy_thres = 130
depth_thres = 150
keypointsNumber = 16

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)  
    plt.show()

def pixel2world(x, fx, fy, ux, uy):
    """
        Converts coordinates from Image coordinates (xyz) to World coordinates (uvd)
    
    """

    x[:, 0] = (x[:, 0] - ux) * x[:, 2] / fx
    x[:, 1] = (x[:, 1] - uy) * x[:, 2] / fy
    return x

def world2pixel(x, fx, fy, ux, uy):
    """
        Converts coordinates from World coordinates (uvd) to Image coordinates (xyz) 
    
    """
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x


def calculateCoM(dpt,minDepth,maxDepth):
    """
    Calculate the center of mass
    :param dpt: depth image
    :return: (x,y,z) center of mass
    """

    dc = dpt.copy()
    dc[dc < minDepth] = 0
    dc[dc > maxDepth] = 0
    cc = ndimage.measurements.center_of_mass(dc > 0)
    num = np.count_nonzero(dc)
    com = np.array((cc[1]*num, cc[0]*num, dc.sum()), np.float)

    if num == 0:
        return np.array((0, 0, 0), np.float)
    else:
        return com/num

def get_center_fast(img, upper=550, lower=1):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    #print centers
    return centers

def get_bbox(centers):
        
    centers = pixel2world(np.array(centers.copy()).reshape((1,3)), fx, fy, u0, v0)

    centerlefttop_test = centers.copy()
    centerlefttop_test[:,0] = centerlefttop_test[:,0]-xy_thres
    centerlefttop_test[:,1] = centerlefttop_test[:,1]+xy_thres

    centerrightbottom_test = centers.copy()
    centerrightbottom_test[:,0] = centerrightbottom_test[:,0]+xy_thres
    centerrightbottom_test[:,1] = centerrightbottom_test[:,1]-xy_thres

    lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)

    Xmin = max(lefttop_pixel[:,0], 0)
    Ymin = max(rightbottom_pixel[:,1], 0)  
    Xmax = min(rightbottom_pixel[:,0], width - 1)
    Ymax = min(lefttop_pixel[:,1], height - 1)

    return (int(Xmin),int(Xmax),int(Ymin),int(Ymax)) # Left Right Top Bottom

def preprocess(img):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    center = get_center_fast(img)
#    width, height = img.shape
    left,right,top,bottom = get_bbox(np.array(center).reshape((1,3)))
    
    imCrop = img.copy()[int(top):int(bottom), int(left):int(right)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize,dtype = 'float32')
#    imgResize = imgResize*-1
    imgResize[np.where(imgResize >= int(center[2]) + depth_thres)] = int(center[2])
    imgResize[np.where(imgResize <= int(center[2]) - depth_thres)] = int(center[2])     
#    
    imgResize = (imgResize - int(center[2]))
    return imgResize, (left,right,top,bottom)

def returnJoints(joints,bbox):
    
    left,right,top,bottom = bbox
    label_xy = np.ones((keypointsNumber, 3), dtype = 'float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
    label_xy[:,0] = (joints[:,0].copy()*(right - left)/cropWidth) + left 
    label_xy[:,1] = (joints[:,1].copy()*(bottom - top)/cropHeight) + top 
    
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0] 
    labelOutputs[:,2] = joints[:,2]
#    labelOutputs = np.asarray(labelOutputs).flatten()
    
    return labelOutputs

def draw_pose(img, pose):
    # Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip, Middle root, Middle mid, Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip.
#    img = input_img.copy()
    sketch = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
                (7, 8), (8, 9), (0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15)]
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, idx , 1)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in sketch:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), idx, 2)
        idx = idx + 1
    #plt.show()
    return img

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

file_prefix = 'image_'

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

frame_block = []
try:
    
    for frame in range(30):
        frames = pipeline.wait_for_frames()
#        aligned_frames = align.process(frames)
        aligned_depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_p,bbox = preprocess(depth_image)
        frame_block.append(depth_p)

    frame_seq= frame_block[-5:]
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

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

        # Preprocess image
        frame_seq.pop(0)
        depth_p,bbox = preprocess(depth_image)
#        print(depth_p.shape)
        frame_seq.append(depth_p)
#        print(frame_seq)
        frame_seqa = np.expand_dims(np.asarray(frame_seq),3)
        frame_seqa = np.expand_dims(frame_seqa,0)
#        print(frame_seq.shape)
#        icvl_net.setInput(cv2.dnn.blobFromImages(frame_seq))
#        joints = icvl_net.forward()
        joints = model.predict(frame_seqa)
        joints = returnJoints(np.reshape(joints,(16,3)),bbox)
#        print(joints)
#        depth_predict = draw_pose(depth_p,joints)
        color_predict = draw_pose(color_image,joints)


        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

#        depth_list = np.dstack([depth_list,depth_image])

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = draw_pose(depth_colormap,joints)

        # Stack both images horizontally
        images = np.hstack((color_predict, depth_colormap))
        
        # Save image, depth and color images
#        image_file = file_prefix + str(idx) + '.png'
#        depth_file = file_prefix + str(idx) + '_depth' + '.png'
#        color_file = file_prefix + str(idx) + '_color' + '.png'
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

