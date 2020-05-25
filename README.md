# Intelligent Hand-Based Rehabilitation System for Stroke Patients

This repository contains a customized tool for automatically detecting and annotating 3D hand joint coordinates based on OpenPose models as well as training and testing scripts for a Convolutional Bi-LSTM model for performing 3D hand pose estimation. 


## OpenPose Multi-View 3D Hand Joint Triangulation Tool

A custom tool based on OpenPose hand detection models for detecting 2D keypoints and automatically triangulating 3D keypoints is included in this repository under the [multicam]() folder. The tool was built upon Intel RealSense APIs and has been customized for Intel RealSense D415 series cameras. The full documentation can be found [here](). 

## Convolutional Bi-Directional LSTM Model (C-Bi-LSTM)

The training and testing scripts for the C-Bi-LSTM model can be found in the main repository folder for the ICVL, NYU and MSRA datasets: **icvl_basic.py**, **nyu_basic**, **msra_basic.py**. There is also a realtime pipeline script that is configured to work with an Intel RealSense D415 series camera: **icvl_realtime_pipeline.py**. The hand centers computed by[ V2V-PoseNet]([https://github.com/mks0601/V2V-PoseNet_RELEASE](https://github.com/mks0601/V2V-PoseNet_RELEASE)) can also be found [here](https://drive.google.com/drive/folders/1-v-VN-eztzoztfHcLt_Y8o5zfRosJ6jt))


Overview of C-Bi-LSTM model:
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/lstm.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/lstm.png)

Demo of the model in realtime deployment: ![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl_demo.gif](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl_demo.gif)

Sample predictions on ICVL and NYU dataset:

![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl1.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl1.png)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl2.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl2.png)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl4.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl4.png)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl5.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl5.png)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl6.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl6.png)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl7.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/icvl7.png)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu1.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu1.png =320x240)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu2.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu2.png =320x240)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu3.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu3.png =320x240)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu4.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu4.png =320x240)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu5.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu5.png =320x240)
![https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu6.png](https://github.com/boonyew/Hand-Joint-Rehab/blob/master/resources/nyu6.png =320x240)