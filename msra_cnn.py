# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:22:23 2020

@author: angbo
"""

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
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
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
cropWidth = 200
cropHeight = 200

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


def loadDepthMap(data_dir,sub,seq,file,labels):
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
        
def generate_data(files,labels,batch_size,val=False):
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
            end_idx = len(file_list)
            file_idx = np.random.randint(0,end_idx)
#            print(len(batch_frames),sub_name,seq_name,file_idx)
            frame_name = file_list[file_idx]
            frame,label = loadDepthMap(base_dir,sub_name,seq_name,frame_name,base_labels)
            batch_frames.append(frame)
            batch_labels.append(label)
                
        image_batch = np.array(batch_frames)
        image_label = np.array(batch_labels)
        image_batch = np.expand_dims(image_batch,3)
        
        yield image_batch,image_label
#		        
#
#x=generate_data(files,labels,10)
#y=next(x)
optmz       = optimizers.Adam(lr=0.001)
def ConvModel():    
    model = Sequential()
    model.add(Conv2D(16, (5,5), activation='relu', padding='same', input_shape=(200,200,1),kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5,5), activation='relu', padding='same',kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(63,activation='relu'))
    model.compile(loss='mse',
                    optimizer=optmz,
                    metrics=['mse','mae'])
    return model
                            # fix random seed for reproducibility
seed        = 29
np.random.seed(seed)


def resLyr(inputs,
            numFilters=16,
            kernelSz=3,
            strides=1,
            activation='relu',
            batchNorm=True,
            convFirst=True,
            lyrName=None):
    convLyr = Conv2D(numFilters,
                    kernel_size=kernelSz,
                    strides=strides,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4),
                    name=lyrName+'_conv' if lyrName else None)
    x = inputs
    if convFirst:
        x = convLyr(x)
        if batchNorm:
            x = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
        if activation is not None:
            x = Activation(activation,name=lyrName+'_'+activation if lyrName else None)(x)
    else:
        if batchNorm:
            x = BatchNormalization(name=lyrName+'_bn' if lyrName else None)(x)
        if activation is not None:
            x = Activation(activation,
                           name=lyrName+'_'+activation if lyrName else None)(x)
        x = convLyr(x)
    return x
        

def resBlkV1(inputs,
             numFilters=16,
             numBlocks=3,
             downsampleOnFirst=True,
             names=None):
    x = inputs
    
    for run in range(0,numBlocks):
        strides = 1
        blkStr = str(run+1)
        if downsampleOnFirst and run == 0:
            strides = 2
        y = resLyr(inputs=x,
                   numFilters=numFilters,
                   strides=strides,
                   lyrName=names+'_Blk'+blkStr+'_Res1' if names else None)
        y = resLyr(inputs=y,
                   numFilters=numFilters,
                   activation=None,
                   lyrName=names+'_Blk'+blkStr+'_Res2' if names else None)
        if downsampleOnFirst and run == 0:
            x = resLyr(inputs=x,
                       numFilters=numFilters,
                       kernelSz=1,
                       strides=strides,
                       activation=None,
                       batchNorm=False,
                       lyrName=names+'_Blk'+blkStr+'_lin' if names else None)
        
        x = add([x,y],
                name=names+'_Blk'+blkStr+'_add' if names else None)
        x = Activation('relu',
                       name=names+'_Blk'+blkStr+'_relu' if names else None)(x)
    return x

frames = 5

def createResNetV1(inputShape=(200,200,1)):
    inputs = Input(shape=inputShape)
    v = resLyr(inputs,
               lyrName='Inpt')
    v = resBlkV1(inputs=v,
                 numFilters=16,
                 numBlocks=3,
                 downsampleOnFirst=False,
                 names='Stg1')
    v = resBlkV1(inputs=v,
                 numFilters=32,
                 numBlocks=3,
                 downsampleOnFirst=True,
                 names='Stg2')
    v = resBlkV1(inputs=v,
                 numFilters=64,
                 numBlocks=3,
                 downsampleOnFirst=True,
                 names='Stg3')

    v = AveragePooling2D(pool_size=8,
                         name='AvgPool')(v)
    v = Flatten()(v)
    
    outputs = Dense(63,
                    activation='relu',
                    kernel_initializer='he_normal')(v)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(loss='mse',
                optimizer=optmz,
                metrics=['mse','mae'])
    return model



modelname   = 'msra_basic'
                            # define the deep learning model

model = createResNetV1()
model.summary()

modelname = 'msra_cnn'
filepath        = modelname + ".hdf5"
checkpoint      = ModelCheckpoint(filepath, 
                                  monitor='val_loss', 
                                  verbose=1, 
                                  save_best_only=True, 
                                  mode='min')

                            # Log the epoch detail into csv
csv_logger      = CSVLogger(modelname +'.csv')
callbacks_list  = [checkpoint,csv_logger]


batch_size = 50

# Debug fitter
#model.fit(x=imgs,y=labels,batch_size=batch_size,epochs=100)

model.fit_generator(
    generate_data(files,labels,batch_size),
    epochs=100,
    validation_data=generate_data(val_files,val_labels,batch_size,val=True),
    validation_steps=17*500//batch_size,
    steps_per_epoch=7*17*500//batch_size,
    verbose=True,
    callbacks=callbacks_list)
#
