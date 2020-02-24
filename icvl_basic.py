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
from PIL import Image
from tqdm import tqdm
import csv
import h5py
import sklearn.metrics as metrics

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

data_dir = r'/home/boonyew/Documents/ICVL/Training/'
val_dir = r'/home/boonyew/Documents/ICVL/Testing/'

center_file = os.path.join(data_dir,'center_train_refined.txt')
label_file = os.path.join(data_dir,'labels.txt')

test_center_file = os.path.join(val_dir,'center_test_refined.txt')
test_label_file = os.path.join(val_dir,'test_seq_1.txt')
#data_dir = r'C:\Users\angbo\Documents\MTech ISS\Capstone\HandJointRehab\ICVL'

fx = 240.99
fy = 240.96
u0 = 160
v0 = 120

height=240
width=320
keypointsNumber = 16
cropWidth = 176
cropHeight = 176
batch_size = 8
xy_thres = 95
depth_thres = 150

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

def loadAnnotations(filename,centers):
    f = open(filename,'r')
    fc = open(centers,'r')
    files = {}
    labels = {}
    for idx,line in enumerate(f):
        if line.rstrip().split(' ') != ['']:
            center = next(fc)
    #        print(center)
            if center.split(' ')[0] == 'invalid':
                pass
            else:
                joints = line.rstrip().split(' ')
                filename = joints[0]
                temp = filename.split('/')
#                print(temp)
                if len(temp) > 2:
                    sub,seq,img = temp
                else:
                    sub = '200'
                    seq,img = temp
    #            joints[-1] = joints[-1].rstrip()
    #            print(joints)
                temp = np.reshape(np.array(joints[1:],dtype='float32'),(keypointsNumber,3))
    #            temp[:,2] = -temp[:,2]
        #            temp = world2pixel(temp,241.42,241.42,160,120)
        #        temp = joints3DToImg(temp)
                center = center.split(' ')
                center[2] = center[2].rstrip()
                center = np.array(center,dtype='float32')
                labels.setdefault(sub,{})
                labels[sub].setdefault(seq,{})
                labels[sub][seq][img] = {}
                labels[sub][seq][img]['labels'] = temp
                labels[sub][seq][img]['center'] = center
                labels[sub][seq][img]['bbox'] = get_bbox(center.reshape((1,3)))
                files.setdefault(sub,{})
                files[sub].setdefault(seq,[])
                files[sub][seq].append(img)
    f.close()
    fc.close()
    return labels,files


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
    Ymin = max(rightbottom_pixel[:,1], 0)  
    Xmax = min(rightbottom_pixel[:,0], width - 1)
    Ymax = min(lefttop_pixel[:,1], height - 1)

    return (int(Xmin),int(Xmax),int(Ymin),int(Ymax)) # Left Right Top Bottom

def loadDepthMap(files,base_dir,sub,seq,file,return_bbox=False):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    filename = os.path.join(base_dir,'Depth',sub,seq,file)
#    print(filename)
#    img = cv2.imread(filename)
    img = np.array(Image.open(filename))
    width, height = img.shape
    left,right,top,bottom = files[sub][seq][file]['bbox']
    joints = files[sub][seq][file]['labels']
    center = files[sub][seq][file]['center']
    
    imCrop = img.copy()[int(top):int(bottom), int(left):int(right)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize,dtype = 'float32')
#    imgResize = imgResize*-1
    imgResize[np.where(imgResize >= int(center[2]) + depth_thres)] = int(center[2])
    imgResize[np.where(imgResize <= int(center[2]) - depth_thres)] = int(center[2])     
#    
    imgResize = (imgResize - int(center[2]))
    
    joints = resizeJoints(joints,left,right,top,bottom)
    
    if return_bbox:
        return imgResize,joints,img,(left,right,top,bottom)
    else:
        return imgResize, joints

def resizeJoints(joints,left,right,top,bottom):
    label_xy = np.ones((keypointsNumber, 3), dtype = 'float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
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

# Debug code
files,filelists = loadAnnotations(label_file,center_file)
test_files,test_filelists = loadAnnotations(test_label_file,test_center_file)
#test_paths = test_files['200']['test_seq_1'].keys()
#test_imgs = [loadDepthMap(test_files,val_dir,'200','test_seq_1',i) for i in list(test_paths)]
#imgs, xlabels = zip(*test_imgs)
#imgs = np.array(imgs)
#xlabels = np.array(xlabels)
#plt.imshow(imgs[10])
#for x,y,z in np.reshape(xlabels[2],(16,3)):
#    plt.plot(x,y,color='green', marker='o')
###-------------------------------------------------------------------------------####
        
frames=5

def generate_data_val(files,filelist,frames,batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    
    sub_idx =0
    seq_idx=0
    file_idx=0
    batch_size=batch_size
    # frames = 5
    while True:
        batch_frames=[]
        batch_labels=[]
        for b in range(batch_size):
            sub_list =list(filelist.keys())
            sub_name = sub_list[sub_idx]
            seq_list = list(filelist[sub_name].keys())
            seq_name =seq_list[seq_idx]
            file_list = filelist[sub_name][seq_name]
            file_list.sort()
            end_idx = len(file_list) - frames
            seq_frames=[]
#            print(len(batch_frames),sub_name,seq_name,file_idx)
            for i in range(frames):
                try:
                    frame_name = file_list[file_idx+i]
                    frame,label = loadDepthMap(files,val_dir,sub_name,seq_name,frame_name)
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
        
def generate_data(files,filelist,frames,batch_size):
    """
        Custom data generator for generating sequences of N Frames for K Batches
    
    """
    batch_size=batch_size
    # frames = 5
    while True:
        batch_frames=[]
        batch_labels=[]
        for b in range(batch_size):
            # Generate random index for sub,seq,start_frame
            sub_list =list(filelist.keys())
            sub_idx = np.random.randint(0,len(sub_list))
            sub_name = sub_list[sub_idx]
            seq_list = list(filelist[sub_name].keys())
            seq_idx = np.random.randint(0,len(seq_list))
            seq_name =seq_list[seq_idx]
            file_list =filelist[sub_name][seq_name]
            file_list.sort()
            factor = np.random.randint(1,2)
            end_idx = len(file_list) - frames*factor
            file_idx = np.random.randint(0,end_idx)
            seq_frames=[]
#            print(len(batch_frames),sub_name,seq_name,file_idx)
            for i in range(frames):
                try:
                    frame_name = file_list[file_idx+i*factor]
                    frame,label = loadDepthMap(files,data_dir,sub_name,seq_name,frame_name)
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
##
#x=generate_data_val(test_files,test_filelists,5,5)
#y=next(x)
##        
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
    model.add(Bidirectional(LSTM(2048,dropout=0.5)))
    # model.add(Dense(256,activation='relu'))
    model.add(Dense(48,activation='relu'))
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

modelname = 'icvl_clstm_basic_120'
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
    generate_data(files,filelists,frames,batch_size),
    epochs=10,
    validation_data=generate_data_val(test_files,test_filelists,frames,batch_size),
    validation_steps=695//batch_size,
    steps_per_epoch=320000//batch_size,
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

def loadModel(modelpath):
    modelGo=LSTMModel()
    modelGo.load_weights(modelpath)
    modelGo.compile(loss='mse', 
                    optimizer=optmz, 
                    metrics=['mse','mae'])
    return modelGo

def testPipeline(modelGo,base_dir,files,filelist,sub,seq,file_idx):
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
    sub_list =list(filelist.keys())
    sub_name = sub_list[sub_idx]
    seq_list = list(filelist[sub_name].keys())
    seq_name =seq_list[seq_idx]
    file_list = filelist[sub_name][seq_name]
    file_list.sort()
    end_idx = len(file_list) - frames
    seq_frames=[]
#            print(len(batch_frames),sub_name,seq_name,file_idx)
    for i in range(frames):
        frame_name = file_list[file_idx+i]
        frame,label, a ,bbox = loadDepthMap(files,base_dir,sub_name,seq_name,frame_name,True)
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
    gtlabel = returnJoints(np.array(label).reshape((16,3)),left,right,top,bottom)
    return img,predict_joints,gtlabel

#modelpath = './result/icvl_clstm_basic_120.hdf5'
#modelGo = loadModel(modelpath)
#
#def testImg(modelGo,sub=0,seq=0,idx=0):
#    
#    x,y,truey = testPipeline(modelGo,val_dir,test_files,test_filelists,sub,seq,idx)
##    img = draw_pose(x,np.reshape(y,(16,3)))
#    plt.imshow(x)
#    for x,y1,z in np.reshape(y,(16,3)):
#        plt.plot(x,y1,color='green', marker='o')
#    for x,y1,z in np.reshape(truey,(16,3)):
#        plt.plot(x,y1,color='red', marker='o')
#    
#testImg(modelGo,0,0,150)


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
