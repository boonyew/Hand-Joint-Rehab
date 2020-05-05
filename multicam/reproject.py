#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:19:42 2020

@author: boonyew
"""


import numpy as np
import cv2
import pptk
import pickle as pkl
import matplotlib.pyplot as plt
fx=628.668
fy=628.668
ux=311.662
vy=231.571

cameras = pkl.load(open('cameras.pkl','rb'))
cams = list(cameras.keys())
intrinsics = np.array([[628.668,0.,311.662],
                       [0.,628.668,231.571],
                       [0.,  0.  ,1.]])
points = pkl.load(open('3dpoints.pkl','rb'))
# Convert to projection matrix
for cam in cameras:
    cameras[cam]['proj'] = np.matmul(intrinsics,cameras[cam]['matrix'][0:3,:])

def world2pixel(x, fx, fy, ux, uy):
    """
        Converts coordinates from World coordinates (uvd) to Image coordinates (xyz) 
    
    """
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x


def find3dpoints(cameras,threshold,undistort=False):
    points3d = {}
    cams = list(cameras.keys())
    frames = len(cameras[cams[0]].keys())-1
    for img in range(frames-1):
        points3d[img] = []
        for jdx in range(21):
            flag,points = find3dpoint(cameras,threshold,img,jdx)
            if flag:
                points3d[img].append(points)
        if len(points3d[img]) >= 21:
            points3d[img] = np.vstack(points3d[img])
        else:
            points3d[img] = 'Invalid Frame'
    return points3d

def find3dpoint(cameras,threshold,img,jdx,undistort=False):
    """Find 3D coordinate using all data given
    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2
    (p.312).
    By default, this function will undistort 2D points before
    finding a 3D point.
    """
    # for info on SVD, see Hartley & Zisserman (2003) p. 593 (see
    # also p. 587)
    # Construct matrices
    A=[]
    for name in cameras:
        cam = cameras[name]
#            if undistort:
#                xy = cam.undistort( [xy] )
        Pmat = cam['proj']
        row2 = Pmat[2,:]
        x,y,c = cam[img][0][jdx]
        if c >= threshold:
            A.append( x*row2 - Pmat[0,:] )
            A.append( y*row2 - Pmat[1,:] )

    # Calculate best point
    if len(A) < 4:
#        print('Invalid point')
        return False,0
    else:
        A=np.array(A)
        u,d,vt=np.linalg.svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize
        return True, X

# Convert to projection matrix
for cam in cameras:
    cameras[cam]['proj'] = np.matmul(intrinsics,cameras[cam]['matrix'][0:3,:])

test = find3dpoints(cameras,0.2)

def draw_pose(pose):
    img = np.zeros((480,640))
    sketch = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, idx, -1)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in sketch:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), 5, 2)
        idx = idx + 1
    #plt.show()
    return img

def show_img(cam_id,frame_id,points):
#    imgs = []
#        rvec2,tvec2 =cv2.solvePnPRansac(test[frame],cameras[cams[cam]][frame][0][:,0:2],intrinsics,None)[1:3]
    rvec = cv2.Rodrigues(cameras[cam_id]['matrix'][0:3,0:3])[0]
    tvec = cameras[cam_id]['matrix'][0:3,3]
    yp = cv2.projectPoints(objectPoints=points[frame_id],rvec=rvec,tvec=tvec,cameraMatrix=intrinsics,distCoeffs=None)[0]
    print(yp)
    print(rvec,tvec)
    img = draw_pose(yp[:,0,:])
#    imgs.append(img)
        
    return img

cam0 = show_img(cams[1],93,points)
