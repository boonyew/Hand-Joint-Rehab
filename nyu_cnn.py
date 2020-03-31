#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import csv
import h5py
import sklearn.metrics as metrics
import scipy.io as scio
import random
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_dir = r'/home/boonyew/Documents/NYU1'
val_dir = r'/home/boonyew/Documents/NYUTest'

center_file = os.path.join(data_dir,'center_train_refined.txt')
label_file = os.path.join(data_dir,'joint_data.mat')

test_center_file = os.path.join(val_dir,'center_test_refined.txt')
test_label_file = os.path.join(val_dir,'joint_data.mat')

breaks_file = os.path.join(data_dir,'breaks.txt')
val_breaks_file = os.path.join(val_dir,'breaks.txt')
#data_dir = r'C:\Users\angbo\Documents\MTech ISS\Capstone\HandJointRehab\ICVL'

fx = 588.03
fy = -587.07
u0 = 320
v0 = 240

# DataHyperParms 
keypointsNumber = 14
cropWidth = 120
cropHeight = 120
batch_size = 8
xy_thres = 110
depth_thres = 150
width=640
height=480
select_joints = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
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

def get_bbox(centers):
        
#    centre_test_world = pixel2world(centers.copy(), fx, fy, u0, v0)

    centerlefttop_test = centers.copy()
    centerlefttop_test[:,0] = centerlefttop_test[:,0]-xy_thres
    centerlefttop_test[:,1] = centerlefttop_test[:,1]+xy_thres

    centerrightbottom_test = centers.copy()
    centerrightbottom_test[:,0] = centerrightbottom_test[:,0]+xy_thres
    centerrightbottom_test[:,1] = centerrightbottom_test[:,1]-xy_thres

    lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)

    Xmin = max(lefttop_pixel[:,0], 0)
    Ymin = max(lefttop_pixel[:,1], 0)  
    Xmax = min(rightbottom_pixel[:,0], width - 1)
    Ymax = min(rightbottom_pixel[:,1], height - 1)

    return (int(Xmin),int(Xmax),int(Ymin),int(Ymax)) # Left Right Top Bottom

def loadDepthMap(base_dir,base_labels,base_bbox,base_centers,idx,return_bbox=False):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    file = 'depth_1_{}.png'.format(str(idx).zfill(7))
    filename = os.path.join(base_dir,file)
#    print(filename)
#    img = cv2.imread(filename)
    img = Image.open(filename)
    r, g, b = img.split()
    r = np.asarray(r, np.int32)
    g = np.asarray(g, np.int32)
    b = np.asarray(b, np.int32)
    dpt = np.bitwise_or(np.left_shift(g, 8), b)
    imgdata = np.asarray(dpt, np.float32)
    idx= max(idx-1,0)
#    width, height, channels = img.shape
#    print(img.shape)
    center = base_centers[idx,:]
    left,right,top,bottom = base_bbox[idx,:]
    
    imCrop = imgdata.copy()[int(top):int(bottom), int(left):int(right)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize,dtype = 'float32')
#    imgResize = imgResize*-1
    imgResize[np.where(imgResize >= int(center[2]) + depth_thres)] = int(center[2])
    imgResize[np.where(imgResize <= int(center[2]) - depth_thres)] = int(center[2])
#    
    imgResize = (imgResize - int(center[2]))
    
    # Normalize image
#    r = np.max(imgResize) - np.min(imgResize)
#    imgResize = imgResize - np.min(imgResize)
#    imgResize = imgResize*255 / r
#    imgResize = imgResize.astype(int)
    joints = resizeJoints(base_labels[:,idx,:].copy(),left,right,top,bottom,center)
    
    if return_bbox:
        return imgResize,joints,img,(left,right,top,bottom)
    else:
        return imgResize, joints

def resizeJoints(joints,left,right,top,bottom,center):
    label_xy = np.ones((keypointsNumber, 3), dtype = 'float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32')
#    joints = pixel2world(joints,fx,fy,u0,v0)
    label_xy[:,0] = (joints[:,0].copy() - left)*cropWidth/(right - left) 
    label_xy[:,1] = (joints[:,1].copy() - top)*cropHeight/(bottom - top) 
    
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0] 
    labelOutputs[:,2] = joints[:,2] - center[2]
#    labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[2])
#    labelOutputs[:,1] = abs(labelOutputs[:,1]-cropHeight)
    labelOutputs = np.asarray(labelOutputs).flatten()
    
    return labelOutputs

def returnJoints(joints,left,right,top,bottom):
    label_xy = np.ones((keypointsNumber, 3), dtype = 'float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
    label_xy[:,0] = (joints[:,0].copy()*(right - left)/cropWidth) + left 
    label_xy[:,1] = (joints[:,1].copy()*(bottom - top)/cropHeight) + top 
    
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0] 
    labelOutputs[:,2] = joints[:,2]
    labelOutputs = np.asarray(labelOutputs).flatten()
    
    return labelOutputs

# Load annotations
full_joints = scio.loadmat(label_file)['joint_uvd'].astype(np.float32)
kjoints = full_joints[0,:,select_joints,:]
centers = np.loadtxt(center_file)
bbox = []
for i in range(centers.shape[0]):
    bboxr = get_bbox(centers[i,:].reshape((1,3)))
    bbox.append(bboxr)
bbox = np.array(bbox)

val_full_joints = scio.loadmat(test_label_file)['joint_uvd'].astype(np.float32)
val_kjoints = val_full_joints[0,:,select_joints,:]
val_centers = np.loadtxt(test_center_file)
val_bbox = []
for i in range(val_centers.shape[0]):
    bboxr = get_bbox(val_centers[i,:].reshape((1,3)))
    val_bbox.append(bboxr)
val_bbox = np.array(val_bbox)

breaks = [int(i.split('_')[2].rstrip()) for i in open(breaks_file)]
invalid_start = [] 
for i in breaks:
    for y in range(1,5):
        invalid_start.append(i - y)
#
        
val_breaks = [int(i.split('_')[2].rstrip()) for i in open(val_breaks_file)]
val_invalid_start = [] 
for i in breaks:
    for y in range(1,5):
        val_invalid_start.append(i - y)
        
train_mean =0 
train_sd = 0
train_imgs = []
train_labels = []

for idx in tqdm(range(1,kjoints.shape[1]+1)):
    img,label = loadDepthMap(data_dir,kjoints,bbox,centers,idx)
    mean_D = img.mean()
    std_D = img.std()
    train_mean += mean_D
    train_sd += std_D
    train_imgs.append(img)
    train_labels.append(label)

train_mean /= 72757
train_sd /= 72757

train_imgs = [(i-train_mean)/train_sd for i in train_imgs]

test_mean = 0
test_sd = 0 

test_imgs = []
test_labels = []

for idx in tqdm(range(1,val_kjoints.shape[1]+1)):
    img,label = loadDepthMap(val_dir,val_kjoints,val_bbox,val_centers,idx)
    mean_D = img.mean()
    std_D = img.std()
    test_mean += mean_D
    test_sd += std_D
    test_imgs.append(img)
    test_labels.append(label)
    
test_mean /= 8252
test_sd /= 8252 
test_imgs = [(i-test_mean)/test_sd for i in test_imgs]
# Debug code
#
#img,label = loadDepthMap(data_dir,kjoints,bbox,centers,6000)
#plt.imshow(train_imgs[20000])
#for x,y,z in np.reshape(train_labels[20000],(14,3)):
#    plt.plot(x,y,color='green', marker='o')
###-------------------------------------------------------------------------------####
        
frames=3

def generate_data(imgs,gtlabels,starts,frames,batch_size,val=False):
    """
        Custom data generator for generating sequences of N Frames for K Batches
    
    """
    batch_size=batch_size
    # frames = 5
    file_idx =0
    end_idx = len(gtlabels) - (frames+1)
#    if val:
#        idx_list = [i for i in range(0,end_idx)]
#    else:
    idx_list = [i for i in range(0,end_idx) if i not in starts]
    if not val:
        random.shuffle(idx_list)
    while True:
        batch_frames=[]
        batch_labels=[]
        for b in range(batch_size):
            if file_idx >= len(idx_list)-1:
                file_idx = 0
            idx = idx_list[file_idx]
#                    frame,label = loadDepthMap(base_dir,base_labels,base_bbox,file_idx+i)
            frame,label = imgs[idx],gtlabels[idx]
            batch_frames.append(frame)
            batch_labels.append(label)
            file_idx += 1
        image_batch = np.array(batch_frames)
        image_label = np.array(batch_labels)
        image_batch = np.expand_dims(image_batch,3)
        
        yield image_batch,image_label
        
#		        
#
#x=generate_data(train_imgs,train_labels,5,10)
##y=next(x)
###
#
#for i in range(72572//10):
#    y=next(x)
#    print(i)
        
def ConvModel2():    
    model = Sequential()
    model.add(Conv2D(32, (7,7), activation='relu', padding='same', input_shape=(cropHeight,cropWidth,1),kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (7,7), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(512, (3,3), activation='relu', padding='same',kernel_initializer='he_normal',
#                    kernel_regularizer=l2(1e-4)))
#    model.add(BatchNormalization())
#    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(42,activation='relu'))
    model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['mse','mae'])
    return model

model = ConvModel2()
model.summary()

def lrSchedule(epoch):
    lr  = 1e-3
    
    if epoch > 10:
        lr  *= 0.5e-3
        
    elif epoch > 8:
        lr  *= 1e-3
        
    elif epoch > 6:
        lr  *= 1e-2
        
    elif epoch > 4:
        lr  *= 1e-1
        
    print('Learning rate: ', lr)
    
    return lr

LRScheduler     = LearningRateScheduler(lrSchedule)

modelname = 'nyu_clstm_cnn'
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]


batch_size = 20

model.fit_generator(
    generate_data(train_imgs,train_labels,invalid_start,frames,batch_size),
    epochs=10,
    validation_data=generate_data(test_imgs,test_labels,val_invalid_start,frames,batch_size,True),
    validation_steps=(8240-frames)//batch_size,
    steps_per_epoch=(72572-frames)//batch_size,
    verbose=True,
    callbacks=callbacks_list)
#
#def test_model():
#    modelGo=LSTMModel()
#    modelGo.load_weights(filepath)
#    modelGo.compile(loss='mse', 
#                    optimizer=optmz, 
#                    metrics=['mse','mae'])
#    val_files,val_labels=loadPaths(val_dir)
#    seqs = val_files['P8']
#    test_imgs = []
#    test_labels = []
#    for i in seqs:
#        imgs = [loadDepthMap(data_dir,'P8',i,x,labels) for x in seqs[i]]
#        imgs, xlabels = zip(*test_imgs)
#        test_imgs.append(imgs)
#        test_labels.append(xlabel)
#    test_imgs = np.array(test_imgs)
#    test_labels = np.array(test_labels)

    # test_paths = files['P0']['1']
    # test_imgs = [loadDepthMap('P0','1',i) for i in test_paths]
    # imgs, xlabels = zip(*test_imgs)
    # imgs = np.array(imgs)
    # xlabels = np.array(xlabels)
    # plt.imshow(imgs[2])
    # for x,y,z in np.reshape(xlabels[2],(21,3)):
    #    plt.plot(x,y,color='green', marker='o')
#
#def loadModel(modelpath):
#    modelGo=LSTMModel()
#    modelGo.load_weights(modelpath)
#    modelGo.compile(loss='mse', 
#                    optimizer=optmz, 
#                    metrics=['mse','mae'])
#    return modelGo
#
#def testPipeline(modelGo,imgs,gtlabels,idx):
#    """
#    Read a depth-map
#    :param filename: file name to load
#    :return: image data of depth image
#    """
#    batch_labels=[]
#    seq_frames = []
#    for i in range(frames):
#        frame,label = imgs[idx+i],gtlabels[idx+i]
#        seq_frames.append(frame)
#        if i == frames-1:
#            batch_labels.append(label)
#            img = frame
#            predict_image = frame
#            left,right,top,bottom = val_bbox[idx+i,:]
#    seq_frames= np.expand_dims(np.array(seq_frames),3)
#    seq_frames = np.expand_dims(seq_frames,0)
#    predictions = modelGo.predict(seq_frames)
#    predictions = predictions.reshape((keypointsNumber,3))
#    predict_joints = returnJoints(predictions,left,right,top,bottom)
#    gtlabel = returnJoints(np.array(label).reshape((14,3)),left,right,top,bottom)
#    return img,predictions,np.array(label).reshape((14,3))
#modelpath = 'nyu_clstm_basic.hdf5'
#modelGo = loadModel(modelpath)
##
##predictions = modelGo.evaluate(generate_data(test_imgs,test_labels,frames,batch_size,True),steps=(8240-frames)//batch_size)
#
#def testImg(modelGo,imgs,gtlabels,idx=0):
#    
#    x,y,truey = testPipeline(modelGo,imgs,gtlabels,idx)
##    img = draw_pose(x,np.reshape(y,(16,3)))
#    plt.imshow(x)
#    for x,y1,z in np.reshape(y,(14,3)):
#        plt.plot(x,y1,color='green', marker='o')
#    for x,y1,z in np.reshape(truey,(14,3)):
#        plt.plot(x,y1,color='red', marker='o')
#    
#testImg(modelGo,test_imgs,test_labels,7996)


def draw_pose(input_img, pose):
    # Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip, Middle root, Middle mid, Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip.
    img = input_img.copy()
    sketch = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
                (7, 8), (8, 9), (0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15)]
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0) , 1)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in sketch:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 255, 0), 2)
        idx = idx + 1
    #plt.show()
    return img
