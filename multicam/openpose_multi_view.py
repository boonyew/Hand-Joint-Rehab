###########################################################################################################################
##                      License: Apache 2.0. See LICENSE file in root directory.                                         ##
###########################################################################################################################
##                  Simple Box Dimensioner with multiple cameras: Main demo file                                         ##
###########################################################################################################################
## Workflow description:                                                                                                 ##
## 1. Place the calibration chessboard object into the field of view of all the realsense cameras.                       ##
##    Update the chessboard parameters in the script in case a different size is chosen.                                 ##
## 2. Start the program.                                                                                                 ##
## 3. Allow calibration to occur and place the desired object ON the calibration object when the program asks for it.    ##
##    Make sure that the object to be measured is not bigger than the calibration object in length and width.            ##
## 4. The length, width and height of the bounding box of the object is then displayed in millimeters.                   ##
###########################################################################################################################

# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np
import pickle as pkl
import sys
import os
from sys import platform
import argparse
import time
import pickle as pkl

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from realsense_device_manager import DeviceManager
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('/home/boonyew/openpose/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
# parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000241.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
# args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/home/boonyew/openpose/models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0

# # Add others in path?
# for i in range(0, len(args[1])):
#     curr_item = args[1][i]
#     if i != len(args[1])-1: next_item = args[1][i+1]
#     else: next_item = "1"
#     if "--" in curr_item and "--" in next_item:
#         key = curr_item.replace('-','')
#         if key not in params:  params[key] = "1"
#     elif "--" in curr_item and "--" not in next_item:
#         key = curr_item.replace('-','')
#         if key not in params: params[key] = next_item

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
datum = op.Datum()

def predict_keypoints(color_image):
    imageToProcess = color_image
    handRectangles = [
        # Left/Right hands person 0
        [
        op.Rectangle(0., 0., 0., 0.),
        op.Rectangle(100., 100., 480., 480.),
        ]
    ]
    
    # Create new datum
    
    datum.cvInputData = imageToProcess
    datum.handRectangles = handRectangles
    
    # Process and display image
    opWrapper.emplaceAndPop([datum])
    # Debug code
    # print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    # print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    # cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)      
    return(datum.handKeypoints[1],datum.cvOutputData)

def calibrateCameras(align,device_manager,frames,chessboard_params):

    """
    1: Calibration
    Calibrate all the available devices to the world co-ordinates.
    For this purpose, a chessboard printout for use with opencv based calibration process is needed.
    
    """

    cameras = {}
#        # Get the intrinsics of the realsense device 
    intrinsics_devices = device_manager.get_device_intrinsics(frames)
    
    # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
    calibrated_device_count = 0
    while calibrated_device_count < len(device_manager._available_devices):
        frames,maps = device_manager.poll_frames(align)
        pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
        transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
        object_point = pose_estimator.get_chessboard_corners_in3d()
        calibrated_device_count = 0
        for device in device_manager._available_devices:
            if not transformation_result_kabsch[device][0]:
                print("Place the chessboard on the plane where the object needs to be detected..")
            else:
                calibrated_device_count += 1

    # Save the transformation object for all devices in an array to use for measurements
    transformation_devices={}
    chessboard_points_cumulative_3d = np.array([-1,-1,-1]).transpose()
    for device in device_manager._available_devices:
        transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
        points3D = object_point[device][2][:,object_point[device][3]]
        points3D = transformation_devices[device].apply_transformation(points3D)
        chessboard_points_cumulative_3d = np.column_stack( (chessboard_points_cumulative_3d,points3D) )
        print(transformation_devices[device].pose_mat)
        cameras[device]={}
        cameras[device]['matrix'] = transformation_devices[device].pose_mat
        np.save('camera_matrix_{}'.format(str(device)),transformation_devices[device].pose_mat)

    # Extract the bounds between which the object's dimensions are needed
    # It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
    chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
    roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)

    print("Calibration completed... \nPlace the box in the field of view of the devices...")

    return cameras

def run_demo():
    
    # Define some constants 
    resolution_width = 640 # pixels
    resolution_height = 480 # pixels
    frame_rate = 15  # fps
    dispose_frames_for_stablisation = 30  # frames
    
    chessboard_width = 6 # squares
    chessboard_height = 9     # squares
    square_size = 0.0253 # meters
    align_to = rs.stream.color
    align = rs.align(align_to)
    try:
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

        # Use the device manager class to enable the devices and get the frames
        device_manager = DeviceManager(rs.context(), rs_config)
        device_manager.enable_all_devices()
        
        # Allow some frames for the auto-exposure controller to stablise
        for frame in range(dispose_frames_for_stablisation):
            frames,maps = device_manager.poll_frames(align)

        assert( len(device_manager._available_devices) > 0 )

        """
        1. Calibrate cameras and return transformation matrix (rotation matrix + translation vectors)

        """

        chessboard_params = [chessboard_height, chessboard_width, square_size] 
        cameras = calibrateCameras(align,device_manager,frames,chessboard_params)

        
        """
        2. Run OpenPose on each view frame

       """

        # Enable the emitter of the devices
        device_manager.enable_emitter(True)

        # Load the JSON settings file in order to enable High Accuracy preset for the realsense
        device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

        # Get the extrinsics of the device to be used later
        extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

        # Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
#        calibration_info_devices = defaultdict(list)
#        for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
#            for key, value in calibration_info.items():
#                calibration_info_devices[key].append(value)
        depth_list = {}
        color_list = {}
        frame_id = 0
        
        # Continue acquisition until terminated with Ctrl+C by the user
        while 1:
             # Get the frames from all the devices
                frames_devices, maps = device_manager.poll_frames(align)
                # print(frames_devices)
                
                # List collector for display
                depth_color= []
                color = []
                devices = [i for i in maps]
                devices.sort()
                
                for i in devices:
                    # 1. Get depth map and colorize
                    temp = maps[i]['depth']
                    depth_list.setdefault(i,[])
                    depth_list[i].append(np.array(temp))
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(temp, alpha=0.03), cv2.COLORMAP_JET)
                    depth_color.append(depth_colormap)
                    
                    # 2. Run OpenPose detector on image
                    joints,img = predict_keypoints(maps[i]['color'])
                    
                    # 3. Save annotated color image for display
                    color.append(img)
                    
                    color_list.setdefault(i,[])
                    color_list[i].append(img)
                    
                    # 4. Save keypoints for that camera viewpoint
                    cameras[i][frame_id] = joints
                    
                frame_id += 1    
                images = np.vstack((np.hstack(color),np.hstack(depth_color)))

                # Show images for debugging
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1)
                
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

    except KeyboardInterrupt:
        print("The program was interupted by the user. Closing the program...")
        
    finally:
        device_manager.disable_streams()
        cv2.destroyAllWindows()
        cam_pkl = open('cameras.pkl','wb')
        pkl.dump(cameras,cam_pkl)
#        for i in keypoints:
#            k = np.array(keypoints[i])
#            np.save('joints_{}.npy'.format(i),k)
#            print(k.shape)
#            d = np.array(depth[i])
#            c = np.array(color_list[i])
#            np.save('depth_{}.npy'.format(i),d)
#            print(d.shape)
#            np.save('color_{}.npy'.format(i),c)
#            print(c.shape)
    
    
if __name__ == "__main__":
    run_demo()
