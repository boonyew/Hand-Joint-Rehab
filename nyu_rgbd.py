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
import pickle as pkl
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
from tensorflow.keras import backend as K
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
    file_rgb = 'rgb_1_{}.png'.format(str(idx).zfill(7))
    filename = os.path.join(base_dir,file)
    filename_rgb = os.path.join(base_dir,file_rgb)
#    print(filename)
#    img = cv2.imread(filename)
    img = Image.open(filename)
    img_rgb = cv2.imread(filename_rgb,0)
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
    
    imCrop = imgdata.copy()[int(top):int(bottom), int(left):int(right)] # image crop
    imCrop_rgb = img_rgb.copy()[int(top):int(bottom), int(left):int(right)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    imgResize = np.asarray(imgResize,dtype = 'float32')
#    imgResize = imgResize*-1
    imgResize[np.where(imgResize >= int(center[2]) + depth_thres)] = int(center[2])
    imgResize[np.where(imgResize <= int(center[2]) - depth_thres)] = int(center[2])
#    
    imgResize = (imgResize - int(center[2]))
    
    imgResize_rgb = np.asarray(cv2.resize(imCrop_rgb, (cropWidth, cropHeight), 
                                          interpolation=cv2.INTER_NEAREST),dtype = 'float32')
    img_rgbd = np.dstack([imgResize,imgResize_rgb])
    # Normalize image
#    r = np.max(imgResize) - np.min(imgResize)
#    imgResize = imgResize - np.min(imgResize)
#    imgResize = imgResize*255 / r
#    imgResize = imgResize.astype(int)
    joints = resizeJoints(base_labels[:,idx,:].copy(),left,right,top,bottom,center)
    
    if return_bbox:
        return img_rgbd,joints,img,(left,right,top,bottom),center
    else:
        return img_rgbd, joints

def resizeJoints(joints,left,right,top,bottom,center):
    label_xy = np.ones((keypointsNumber, 3), dtype = 'float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32')
#    joints = pixel2world(joints,fx,fy,u0,v0)
    label_xy[:,0] = (joints[:,0].copy() - left)*cropWidth/(right - left) 
    label_xy[:,1] = (joints[:,1].copy() - top)*cropHeight/(bottom - top) 
    
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0] 
    labelOutputs[:,2] = joints[:,2]  - center[2]
#    labelOutputs[:,1] = abs(labelOutputs[:,1]-cropHeight)
    labelOutputs = np.asarray(labelOutputs).flatten()
    
    return labelOutputs

def returnJoints(joints,left,right,top,bottom,center):
    label_xy = np.ones((keypointsNumber, 3), dtype = 'float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
    label_xy[:,0] = (joints[:,0].copy()*(right - left)/cropWidth) + left 
    label_xy[:,1] = (joints[:,1].copy()*(bottom - top)/cropHeight) + top 
    
    labelOutputs[:,1] = label_xy[:,1]
    labelOutputs[:,0] = label_xy[:,0] 
    labelOutputs[:,2] = joints[:,2]+ center[2]
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
for i in val_breaks:
    for y in range(1,5):
        val_invalid_start.append(i - y)
#        
#train_mean =0 
#train_sd = 0
        
#train_imgs = []
#train_labels = []
#
#for idx in tqdm(range(1,kjoints.shape[1]+1)):
#    img,label = loadDepthMap(data_dir,kjoints,bbox,centers,idx)
##    mean_D = img.mean()
##    std_D = img.std()
##    train_mean += mean_D
##    train_sd += std_D
#    train_imgs.append(img)
#    train_labels.append(label)
#
##train_mean /= 72757
##train_sd /= 72757
##
##train_imgs = [(i-train_mean)/train_sd for i in train_imgs]
##
##test_mean = 0
##test_sd = 0 
#
#test_imgs = []
#test_labels = []
#
#for idx in tqdm(range(1,val_kjoints.shape[1]+1)):
#    img,label = loadDepthMap(val_dir,val_kjoints,val_bbox,val_centers,idx)
##    mean_D = img.mean()
##    std_D = img.std()
##    test_mean += mean_D
##    test_sd += std_D
#    test_imgs.append(img)
#    test_labels.append(label)
    
#test_mean /= 8252
#test_sd /= 8252 
#test_imgs = [(i-test_mean)/test_sd for i in test_imgs]

train_imgs = pkl.load(open('nyu_train_imgs.pkl','rb'))
train_labels = pkl.load(open('nyu_train_labels.pkl','rb'))
test_imgs = pkl.load(open('nyu_test_imgs.pkl','rb'))
test_labels = pkl.load(open('nyu_test_labels.pkl','rb'))

test_seqs = []
temp_seq = []
for idx in range(1,len(test_imgs)+1):
    if idx in val_breaks:
        test_seqs.append(temp_seq)
        temp_seq = []
        temp_seq.append(idx)
    else:
        temp_seq.append(idx)
test_seqs.append(temp_seq)
        
    

# Debug code
#
#img,label = loadDepthMap(data_dir,kjoints,bbox,centers,6000)
#plt.imshow(img)
#for x,y,z in np.reshape(train_labels[20000],(14,3)):
#    plt.plot(x,y,color='green', marker='o')
###-------------------------------------------------------------------------------####
        
frames=5

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
            seq_frames = []
            for i in range(frames):
                try:
#                    frame,label = loadDepthMap(base_dir,base_labels,base_bbox,file_idx+i)
                    frame,label = imgs[idx+i],gtlabels[idx+i]
                    seq_frames.append(frame)
                    if i == frames-1:
                        
                        batch_labels.append(label)
                except:
#                    seq_frames.append(seq_frames[-1])
                    break
            file_idx += 1
            if seq_frames:
                seq_frames= np.array(seq_frames)
                batch_frames.append(seq_frames)
#            print(len(batch_frames),file_idx,idx)
        image_batch = np.array(batch_frames)
        image_label = np.array(batch_labels)
#        image_batch = np.expand_dims(image_batch,4)
        
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

def ConvModel():    
    model = Sequential()
    model.add(Conv2D(16, (7,7), activation='relu', padding='same', input_shape=(cropHeight,cropWidth,2),kernel_initializer='he_normal',
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
def mean_joint(y_true,y_pred):
    y_pred = K.reshape(y_pred, (140,3))
    y_true = K.reshape(y_true, (140,3))
    return K.mean(K.sum(abs(y_true-y_pred),axis=1))

def LSTMModel():
    model = Sequential()
    model.add(TimeDistributed(ConvModel(),input_shape=(frames,cropHeight, cropWidth,2)))
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
    model.add(Bidirectional(LSTM(1024,dropout=0.5)))
    # model.add(Dense(256,activation='relu'))
    model.add(Dense(42,activation='relu'))
    model.compile(loss=mean_joint,
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

modelname = 'nyu_clstm_rgbd_new_fix'
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
    generate_data(train_imgs,train_labels,invalid_start,frames,batch_size),
    epochs=10,
    validation_data=generate_data(test_imgs,test_labels,val_invalid_start,frames,batch_size,True),
    validation_steps=(8240-frames)//batch_size,
    steps_per_epoch=(72572-frames)//batch_size,
    verbose=True,
    callbacks=callbacks_list)


#### TEST SCRIPT####

def loadModel(modelpath):
    modelGo=LSTMModel()
    modelGo.load_weights(modelpath)
    modelGo.compile(loss=mean_joint, 
                    optimizer=optmz, 
                    metrics=['mse','mae'])
    return modelGo

def testPipeline(modelGo,imgs,gtlabels,seq,file_idx):
    """
    Read a depth-map
    :param filename: file name to load
    :return: image data of depth image
    """
    frames = 5
#            print(len(batch_frames),sub_name,seq_name,file_idx)
    seq_frames=[]
    for i in range(frames):
        if file_idx +i < 0:
            file_id = 0
        else:
            file_id = file_idx +i 
        frame_name = seq[file_id]
        frame,label, a ,bbox,c = loadDepthMap(val_dir,val_kjoints,val_bbox,val_centers,frame_name,True)
        seq_frames.append(frame)
        if i == frames-1:
            img = a
#            img[np.where(img >= 1000)] =0
            predict_image = frame
            left,right,top,bottom = bbox
            center = c
#    seq_frames= np.expand_dims(np.array(seq_frames),3)
    seq_frames = np.expand_dims(seq_frames,0)
    predictions = modelGo.predict(seq_frames)
    predictions = predictions.reshape((keypointsNumber,3))
    predict_joints = returnJoints(predictions,left,right,top,bottom,center)
    gtlabel = returnJoints(np.array(label).reshape((14,3)),left,right,top,bottom,center)
    return img,predict_joints,gtlabel
#
modelpath = './result/nyu_clstm_rgbd_new_fix.hdf5'
modelGo = loadModel(modelpath)

test_predict_labels = []
gt_labels = []

for seq in test_seqs:
    seq_len = len(seq)
    for idx in range(-4,seq_len-4):
        x,y,truey = testPipeline(modelGo,test_imgs,test_labels,seq,idx)
        test_predict_labels.append(y)
        gt_labels.append(truey)
np.savetxt('nyu_predict.txt',np.array(test_predict_labels),fmt='%.3f')

from multicam.util import draw_pose,draw_angles

x,y,truey = testPipeline(modelGo,test_imgs,test_labels,seq,100)
img = draw_pose(y.reshape((14,3)),'nyu',np.array(x))
plt.imshow(img)
#cv2.imwrite('icvl7.png',img)

test_predict_labels = np.array(test_predict_labels).reshape((1596,16,3))

test_angles = []
for i in test_predict_labels:
    temp = list(util.draw_angles(i,'icvl').values())
    test_angles.append(temp)

test_angles = np.array(test_angles)

gt_labels = np.array(gt_labels).reshape((1596,16,3))

gt_angles = []
for i in gt_labels:
    temp = list(draw_angles(i,'nyu').values())
    gt_angles.append(temp)

gt_angles = np.array(gt_angles)

np.mean(abs(test_angles - gt_angles))