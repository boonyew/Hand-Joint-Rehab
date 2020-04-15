# OpenPose Multi-View Hand Pose Triangulation

---

# 1. Installation and Setup

To use the OpenPose Multi-View triangulation tool, the official OpenPose system needs to be installed as well as the necessary Python API bindings. The system is also build for Intel RealSense cameras, which will require the RealSense drivers for Windows and Ubuntu/Other Linux distros. 

---
## 1.1 Install OpenPose

1.  Install the OpenPose prerequisites, which can be found in the [prerequisites guide](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/prerequisites.md)
2. Install OpenPose from source using the official guide [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#installation). For Ubuntu or other Linux OS, CMake version 3.15 and above is required. 

## 1.2 Install Python Requirements

1. Git clone the repository and run `cd ./Hand-Joint-Rehab/multicam`
2. Install the required Python3 packages in `requirements.txt`

---
# 2. Environment Setup

The system requires at least two cameras to retrieve 3D points, but will work fine with a single camera for gathering 2D points. A printout of the chessboard.png for calibration is also required (A4 size should be good). An example of the layout of the cameras and chessboard printout is show below:


---

# 3. Usage

There are three steps involved in using this system: **1. Camera Calibration** and **2. 2D Hand Keypoint Detection** and **3. 3D Keypoint Triangulation**. 

In order to start using the tool, open a terminal in the `multicam` folder and run the following command:
`python3 openpose_multiview.py`. 

By default, the flags for recording the RGB and Depth and hand bounding boxes are turned off. In order to enable them, run the following command: `python3 openpose_multiview.py y y`

The details of the steps involved are discussed below.

---

## 3.1 Camera Calibration

The objective of this step is to calibrate the RealSense cameras, and to retreive their transformation matrices which will be used in Step 3 where the multiple 2D points are triangulated. In this step, the chessboard piece has to be visible to the cameras.

Once the script has been run, the calibration process will start and the following output will appear in the terminal:

`Place the chessboard on the plane where the object needs to be detected..`
`Place the chessboard on the plane where the object needs to be detected..`

During this step, the calibration process will continue until a satisfactory estimation of the camera parameters is achieved. You may have to adjust the chessboard around until the following output is achieved:

`Calibration completed... 
Place your hand in the field of view of the devices...`

### 3.11 Troubleshooting

At times the calibration process may fail to estimate the camera parameters, you can try the following methods to solve it:

1. Ensure that the entire chessboard is visible from the cameras' point of views/images
2. Ensure that the surroundings are in reasonably good lighting and without shadows being cast onto the printout

## 3.2 2D Hand Keypoint Detection

The 2D hand keypoint detection is done using the OpenPose detector models for each of the RGB images retreived from the camera. By default, the bounding box supplied is a square box in the center of the image. For best results, the cameras can be adjusted to ensure that the hand is in the middle of each of the images captured by the cameras.

To use custom bounding box coordinates:

1. In order to supply new bounding box coordinates, the `FLAGS_USE_BBOX` argument has to be enabled by running `python3 openpose_multiview.py n y`

2. The bounding box coordinates can by adding the respective coordinates to the `bbox` object:
`bbox = {}`
`bbox['821212062729'] = [155.,60.,350.,350.]`
`bbox['851112060943'] = [100.,40.,350.,350.]`
`bbox['851112062097'] = [140.,25.,350.,350.]`

## 3.3 3D Keypoint Triangulation

By default, the 3D keypoint triangulation is performed using the detected keypoints from the 2D hand keypoint detection step. A threshold is set in order to filter poorly detected keypoints from that step in order to prevent invalid training labels and is currently set at **0.2**

This threshold can be modified in the following line:

`#Triangulate 3d keypoints`
`points[frame_id] = find3dpoints(cameras,0.2,frame_id)`

