
import numpy as np
import cv2
import pptk
import pickle as pkl
import matplotlib.pyplot as plt
import vg
from enum import Enum

def vec_from_points(p1,p2):
    v1 = []
    for i in range(len(p1)):
        temp = p2[i] - p1[i]
        v1.append(temp)
    return np.array(v1)

def find_angle(p1,p2,p3):
    v1 = vec_from_points(p1,p2)
    v2 = vec_from_points(p3,p2)
    angle = vg.angle(v1,v2)
    return angle

def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    elif dataset == 'openpose':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]


def get_angle_setting(dataset):
    if dataset == 'icvl':
        return  {2:[1,3], 3:[2,4],5:[0,6], 6:[5,7], 7:[6,8], 9:[0,10],
                10:[9,11],11:[10,12],13:[0,14],14:[13,15],15:[14,16],
                17:[0,18],18:[17,19],19:[18,20]}
    elif dataset == 'nyu':
        return  {2:[1,3], 3:[2,4],5:[0,6], 6:[5,7], 7:[6,8], 9:[0,10],
                10:[9,11],11:[10,12],13:[0,14],14:[13,15],15:[14,16],
                17:[0,18],18:[17,19],19:[18,20]}        
    elif dataset == 'msra':
        return  {2:[1,3], 3:[2,4],5:[0,6], 6:[5,7], 7:[6,8], 9:[0,10],
                10:[9,11],11:[10,12],13:[0,14],14:[13,15],15:[14,16],
                17:[0,18],18:[17,19],19:[18,20]}              
    elif dataset == 'openpose':
        return  {2:[1,3], 3:[2,4],5:[0,6], 6:[5,7], 7:[6,8], 9:[0,10],
                10:[9,11],11:[10,12],13:[0,14],14:[13,15],15:[14,16],
                17:[0,18],18:[17,19],19:[18,20]}         

def get_sketch_color(dataset):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    if dataset == 'icvl':
        return [RED, RED, RED, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE]
    elif dataset == 'nyu':
        return (GREEN, RED, PURPLE, YELLOW, BLUE, BLUE, GREEN,
                RED, PURPLE, YELLOW, BLUE, CYAN, CYAN)
    elif dataset == 'msra':
        return [RED, RED, RED, RED, GREEN, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE, PURPLE]
    elif dataset == 'openpose':
        return [GREEN, BLUE, YELLOW, PURPLE, RED,
              GREEN, GREEN, GREEN,
              BLUE, BLUE, BLUE,
              YELLOW, YELLOW, YELLOW,
              PURPLE, PURPLE, PURPLE,
              RED, RED, RED]

def get_joint_color(dataset):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244)
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    if dataset == 'icvl':
        return [CYAN, RED, RED, RED, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE]
    elif dataset == 'nyu':
        return (GREEN, GREEN, RED, RED, PURPLE, PURPLE, YELLOW, YELLOW,
                BLUE, BLUE, BLUE,
                CYAN, CYAN, CYAN)
    elif dataset == 'msra':
        return [CYAN, RED, RED, RED, RED, GREEN, GREEN, GREEN,
                GREEN,
                BLUE, BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW, YELLOW,
                PURPLE, PURPLE, PURPLE, PURPLE]
    elif dataset == 'openpose':
        return [CYAN, GREEN, BLUE, YELLOW, PURPLE, RED, GREEN, GREEN, GREEN,
                BLUE, BLUE, BLUE, YELLOW, YELLOW, YELLOW, PURPLE, PURPLE, PURPLE,
                RED, RED, RED]

def find3dpoints_rt(cameras,threshold,img,undistort=False):
    points3d = []
    cams = list(cameras.keys())
    for jdx in range(21):
        flag,points = find3dpoint(cameras,threshold,img,jdx)
        if flag:
            points3d.append(points)
    if len(points3d) < 21:
        points3d = 'Invalid Frame'
    else:
        points3d = np.array(points3d)
    return points3d

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
    try:
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
    except:
        return False,0

def show_img(cameras,device,frame_id,points):
#    imgs = []
#        rvec2,tvec2 =cv2.solvePnPRansac(test[frame],cameras[cams[cam]][frame][0][:,0:2],intrinsics,None)[1:3]
    rvec = cv2.Rodrigues(cameras[device]['matrix'][0:3,0:3])[0]
    tvec = cameras[device]['matrix'][0:3,3]
    if points[frame_id] == 'Invalid Frame':
        img = np.zeros((480,640))
    else:
        yp = cv2.projectPoints(objectPoints=points[frame_id],rvec=rvec,tvec=tvec,cameraMatrix=intrinsics,distCoeffs=None)[0]
        img = draw_pose(yp[:,0,:])
#    imgs.append(img)
    return img

def draw_angles(points,dataset):
    angle_joints = get_angle_setting(dataset)
    angles = {}
    for idx,pt in enumerate(points):
        if idx in angle_joints.keys():
            p1,p3 = angle_joints[idx]
            theta = find_angle(points[p1],pt,points[p3])
            angles[idx] = round(theta,1)
            print(round(theta,1))
    return angles

def draw_pose(pose,points,dataset):
    img = np.zeros((480,640))
    sketch = get_sketch_setting(dataset)
    angles = draw_angles(points,dataset)
    idx = 0
    #plt.figure()
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, idx, -1)
        if idx in angles.keys():
            cv2.putText(img,str(angles[idx]),(int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (63,63,63),
                        1,
                        cv2.LINE_AA)
        #plt.scatter(pt[0], pt[1], pt[2])
        idx = idx + 1
    idx = 0
    for x, y in sketch:
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), 5, 2)
        idx = idx + 1
    #plt.show()
    return img