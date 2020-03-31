#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:36:22 2020

@author: astroboon
"""

import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
import csv
import h5py
import sklearn.metrics as metrics
import struct
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model,Sequential,load_model
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

data_dir = r'/home/boonyew/Documents/MSRA/train'
val_dir = r'/home/boonyew/Documents/MSRA/test'
center_dir = r'/home/boonyew/Documents/MSRA/center'
keypoints_num = 21
test_subject_id = 3
cubic_size = 200
cropWidth = 120
cropHeight = 120
keypointsNumber=21
def joints3DToImg(sample):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = np.zeros((sample.shape[0], 3), np.float32)
    for i in range(sample.shape[0]):
        ret[i] = joint3DToImg(sample[i])
    return ret

def joint3DToImg(sample):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    fx, fy, ux, uy = 241.42,241.42,160,120
    ret = np.zeros((3, ), np.float32)
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0]/sample[2]*fx+ux
    ret[1] = uy-sample[1]/sample[2]*fy
    ret[2] = sample[2]
    return ret


def loadDepthMap(data_dir,sub,seq,file,labels,return_bbox=False):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    filename=os.path.join(data_dir,sub,seq,file)
    with open(filename, 'rb') as f:
        # first 6 uint define the full image
        width = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        left = struct.unpack('i', f.read(4))[0]
        top = struct.unpack('i', f.read(4))[0]
        right = struct.unpack('i', f.read(4))[0]
        bottom = struct.unpack('i', f.read(4))[0]
        patch = np.fromfile(f, dtype='float32', sep="")
#        print(patch.shape)
        imgdata = np.zeros((height, width), dtype='float32')
        imgdata[top:bottom, left:right] = patch.reshape([bottom-top, right-left])
        
        imCrop = imgdata.copy()[int(top):int(bottom), int(left):int(right)]
        imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
        imgResize = np.asarray(imgResize,dtype = 'float32') 
        joints = labels[sub][seq][file]
        joints = resizeJoints(joints,left,right,top,bottom)
        
    if return_bbox:
        return imgResize,joints,imgdata,(left,right,top,bottom)
    else:
        return imgResize, joints

def resizeJoints(joints,left,right,top,bottom):
        ## label
    label_xy = np.ones((21, 3), dtype = 'float32') 
    labelOutputs = np.ones((21, 3), dtype = 'float32') 
    label_xy[:,0] = (joints[:,0].copy() - left)*cropWidth/(right - left) 
    label_xy[:,1] = (joints[:,1].copy() - top)*cropHeight/(bottom - top) 
    
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0] 
    labelOutputs[:,2] = joints[:,2]
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
    
def loadJoints(filename):
    f = open(filename,'r')
    labels = []
    for idx,line in enumerate(f):
        if idx == 0:
            pass
        else:
            joints = line.split(' ')
            joints[-1] = joints[-1].rstrip()
            temp = np.reshape(np.array(joints,dtype='float32'),(21,3))
            temp[:,2] = -temp[:,2]
#            temp = world2pixel(temp,241.42,241.42,160,120)
            temp = joints3DToImg(temp)
            labels.append(temp)
    return np.array(labels)

def loadPaths(data_dir):
    file_dir = {}
    labels = {}
    for sub in os.listdir(data_dir):
        if sub[0] == 'P':
            file_dir[sub] = {}
            labels[sub] = {}
            sub_path = os.path.join(data_dir,sub)
            for seq in os.listdir(sub_path):
                labels[sub][seq] = {}
                seq_path = os.path.join(sub_path,seq)
                file_dir[sub][seq] = [file for file in os.listdir(seq_path) if file.split('.')[1] == 'bin']
                file_dir[sub][seq].sort()
                joints_path = os.path.join(seq_path,'joint.txt')
                joints = loadJoints(joints_path)
                for idx,file in enumerate(file_dir[sub][seq]):
                    labels[sub][seq][file] = joints[idx]
    return file_dir,labels

# Debug code
files, labels = loadPaths(data_dir)
val_files,val_labels=loadPaths(val_dir)
#test_paths = files['P0']['1']
#test_imgs = [loadDepthMap('P0','1',i) for i in test_paths]
#imgs, xlabels = zip(*test_imgs)
#imgs = np.array(imgs)
#xlabels = np.array(xlabels)
#plt.imshow(imgs[2])
#for x,y,z in np.reshape(xlabels[2],(21,3)):
#    plt.plot(x,y,color='green', marker='o')
###-------------------------------------------------------------------------------####
        
frames=5

def generate_data_val(files,labels,frames,batch_size,val=False):
    """Replaces Keras' native ImageDataGenerator."""
    if val:
        base_dir = val_dir
        base_labels = val_labels
    else: 
        base_dir = data_dir
        base_labels = labels
    sub_idx =0
    seq_idx=0
    file_idx=0
    batch_size=batch_size
    # frames = 5
    while True:
        batch_frames=[]
        batch_labels=[]
        for b in range(batch_size):
            sub_list =list(files.keys())
            sub_name = sub_list[sub_idx]
            seq_list = list(files[sub_name].keys())
            seq_name =seq_list[seq_idx]
            file_list = files[sub_name][seq_name]
            file_list.sort()
            end_idx = len(file_list) - frames
            seq_frames=[]
#            print(len(batch_frames),sub_name,seq_name,file_idx)
            for i in range(frames):
                try:
                    frame_name = file_list[file_idx+i]
                    frame,label = loadDepthMap(base_dir,sub_name,seq_name,frame_name,base_labels)
                    seq_frames.append(frame)
                    if i == frames-1:
                        batch_labels.append(label)
                except:
                    seq_frames = None
                    break
            file_idx +=1
            if seq_frames:
                seq_frames= np.array(seq_frames)
                batch_frames.append(seq_frames)
            if file_idx >= end_idx+1:
                if seq_idx  >= len(seq_list)-1:
                    if sub_idx >= len(sub_list)-1:
                        sub_idx = 0
                    else:
                        sub_idx +=1
                    seq_idx =0
                else:	
                    seq_idx +=1
                file_idx = 0
                
        image_batch = np.array(batch_frames)
        image_label = np.array(batch_labels)
        image_batch = np.expand_dims(image_batch,4)
        
        yield image_batch,image_label
        
def generate_data(files,labels,frames,batch_size,val=False):
    """
        Custom data generator for generating sequences of N Frames for K Batches
    
    """
    if val:
        base_dir = val_dir
        base_labels = val_labels
    else: 
        base_dir = data_dir
        base_labels = labels
    batch_size=batch_size
    # frames = 5
    while True:
        batch_frames=[]
        batch_labels=[]
        for b in range(batch_size):
            # Generate random index for sub,seq,start_frame
            sub_list =list(files.keys())
            sub_idx = np.random.randint(0,len(sub_list))
            sub_name = sub_list[sub_idx]
            seq_list = list(files[sub_name].keys())
            seq_idx = np.random.randint(0,len(seq_list))
            seq_name =seq_list[seq_idx]
            file_list = files[sub_name][seq_name]
            file_list.sort()
            end_idx = len(file_list) - frames
            file_idx = np.random.randint(0,end_idx)
            seq_frames=[]
#            print(len(batch_frames),sub_name,seq_name,file_idx)
            for i in range(frames):
                try:
                    frame_name = file_list[file_idx+i]
                    frame,label = loadDepthMap(base_dir,sub_name,seq_name,frame_name,base_labels)
                    seq_frames.append(frame)
                    if i == frames-1:
                        batch_labels.append(label)
                except:
                    seq_frames = None
                    break
            if seq_frames:
                seq_frames= np.array(seq_frames)
                batch_frames.append(seq_frames)
                
        image_batch = np.array(batch_frames)
        image_label = np.array(batch_labels)
        image_batch = np.expand_dims(image_batch,4)
        
        yield image_batch,image_label
#		        
#
#x=generate_data(files,labels,10)
#y=next(x)
        
def ConvModel():    
    model = Sequential()
    model.add(Conv2D(16, (7,7), activation='relu', padding='same', input_shape=(cropHeight,cropWidth,1),kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (7,7), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dense(256,activation='relu'))
    return model
                            # fix random seed for reproducibility
seed        = 29
np.random.seed(seed)
optmz       = optimizers.Adam(lr=0.001)
                            # define the deep learning model

def LSTMModel():
    model = Sequential()
    model.add(TimeDistributed(ConvModel(),input_shape=(frames,cropHeight, cropWidth,1)))
#    model.add(LSTM(2048,
#         return_sequences=True,
#         dropout=0.25,
#         recurrent_dropout=0.25))
#    model.add(LSTM(512,
#         return_sequences=True,
#         dropout=0.25,
#         recurrent_dropout=0.25))
    # model.add(LSTM(2048,
    #         dropout=0.5))
    model.add(Bidirectional(LSTM(1536,dropout=0.5)))
    # model.add(Dense(256,activation='relu'))
    model.add(Dense(63,activation='relu'))
    model.compile(loss='mse',
                    optimizer=optmz,
                    metrics=['mse','mae'])
    return model

model = LSTMModel()
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

modelname = 'msra_clstm_basic_120_P6'
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_loss', 
                                  verbose=0, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]


batch_size = 10

model.fit_generator(
    generate_data(files,labels,frames,batch_size),
    epochs=10,
    validation_data=generate_data_val(val_files,val_labels,frames,batch_size,val=True),
    validation_steps=17*(500-frames)//batch_size,
    steps_per_epoch=7*17*(500-frames)//batch_size,
    verbose=True,
    callbacks=callbacks_list)
#


def loadModel(modelpath):
    modelGo=LSTMModel()
    modelGo.load_weights(modelpath)
    modelGo.compile(loss='mse', 
                    optimizer=optmz, 
                    metrics=['mse','mae'])
    return modelGo

def testPipeline(modelGo,base_dir,files,sub,seq,file_idx):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """

    sub_idx =sub
    seq_idx=seq
    file_idx=file_idx
    frames = 5
    batch_labels=[]
    sub_list =list(files.keys())
    sub_name = sub_list[sub_idx]
    seq_list = list(files[sub_name].keys())
    seq_name =seq_list[seq_idx]
    file_list = files[sub_name][seq_name]
    file_list.sort()
    end_idx = len(file_list) - frames
    seq_frames=[]
#            print(len(batch_frames),sub_name,seq_name,file_idx)
    for i in range(frames):
        frame_name = file_list[file_idx+i]
        frame,label, a ,bbox = loadDepthMap(val_dir,sub_name,seq_name,frame_name,val_labels,True)
        seq_frames.append(frame)
        if i == frames-1:
            batch_labels.append(label)
            img = a
            predict_image = frame
            left,right,top,bottom = bbox
    seq_frames= np.expand_dims(np.array(seq_frames),3)
    seq_frames = np.expand_dims(seq_frames,0)
    predictions = modelGo.predict(seq_frames)
    predictions = predictions.reshape((keypointsNumber,3))
    predict_joints = returnJoints(predictions,left,right,top,bottom)
    gtlabel = returnJoints(np.array(label).reshape((keypointsNumber,3)),left,right,top,bottom)
    return img,predict_joints,gtlabel
#
#modelpath = './result/msra_clstm_basic_120_P6.hdf5'
#model = loadModel(modelpath)
#
#def testImg(modelGo,sub=0,seq=0,idx=0):
#    val_files,val_labels,frames,batch_size,
#    x,y,truey = testPipeline(modelGo,val_dir,val_files,sub,seq,idx)
#    plt.imshow(x)
#    for x,y1,z in np.reshape(y,(21,3)):
#        plt.plot(x,y1,color='green', marker='o')
#    for x,y1,z in np.reshape(truey,(21,3)):
#        plt.plot(x,y1,color='red', marker='o')
#    
#testImg(modelGo,0,0,200)

def draw_pose(input_img, pose):
    # Palm, Thumb root, Thumb mid, Thumb tip, Index root, Index mid, Index tip, Middle root, Middle mid, Middle tip, Ring root, Ring mid, Ring tip, Pinky root, Pinky mid, Pinky tip.
    img = input_img.copy()
    sketch = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, 5, -1)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in sketch:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), 5, 2)
        idx = idx + 1
    #plt.show()
    return img
