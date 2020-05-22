# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import pyrealsense2 as rs
import numpy as np
import cv2

import argparse
from os import makedirs
from os.path import exists, join
import shutil
import json
from enum import IntEnum

import csv

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            makedirs(path_folder)
        else:
            exit()

def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Realsense Recorder. Please select one of the optional arguments")

    parser.add_argument("--output_folder",
                        default='dataset/realsense/',
                        help="set output folder")

    parser.add_argument("--record_rosbag",
                        action='store_true',
                        help="Recording rgbd stream into realsense.bag")

    parser.add_argument("--record_imgs",
                        action='store_true',
                        help="Recording save color and depth images into realsense folder")

    parser.add_argument("--playback_rosbag",
                        action='store_true',
                        help="Play recorded realsense.bag file")

    args = parser.parse_args()

    # Hardcoding
    args.record_imgs = True
    args.record_rosbag = False

    if sum(o is not False for o in vars(args).values()) != 2:
        parser.print_help()
        exit()

    path_output = args.output_folder
    path_depth = join(args.output_folder, "depth")
    path_color = join(args.output_folder, "color")
    if args.record_imgs:
        make_clean_folder(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)

    path_bag = join(args.output_folder, "realsense.bag")
    if args.record_rosbag:
        if exists(path_bag):
            user_input = input("%s exists. Overwrite? (y/n) : " % path_bag)
            if user_input.lower() == 'n':
                exit()

    # Configure D435 + T265
    ctx = rs.context()
    devices = ctx.query_devices()
    pipelines = []
    for device in devices[::-1]:
        if "D435" in str(device):
            print("D435 detected")
            pipeline_D435 = rs.pipeline(ctx)
            config_D435 = rs.config()
            config_D435.enable_device(device.get_info(rs.camera_info.serial_number))
            # note: using 640 x 480 depth resolution produces smooth depth boundaries
            # using rs.format.bgr8 for color image format for OpenCV based image visualization
            config_D435.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config_D435.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #640, 480 or 1280, 720
            profile = pipeline_D435.start(config_D435)
            pipelines.append(pipeline_D435)

        elif "T265" in str(device):
            print("T265 detected")
            pipeline_T265 = rs.pipeline(ctx)
            config_T265 = rs.config()
            config_T265.enable_device(device.get_info(rs.camera_info.serial_number))
            config_T265.enable_stream(rs.stream.pose)
            pipeline_T265.start(config_T265)
            pipelines.append(pipeline_T265)
        
        else:
            print("No device detected. Please plug one!")


    # Start streaming
    depth_sensor = profile.get_device().first_depth_sensor()

    laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    print("laser power = ", laser_pwr)
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    print("laser power range = " , laser_range.min , "~", laser_range.max)
    # set_laser = 0
    # #just simply add 10 to test set function
    # if laser_pwr + 10 > laser_range.max:
    #     set_laser = laser_range.max
    # else:
    #     set_laser = laser_pwr + 10
    depth_sensor.set_option(rs.option.laser_power, laser_range.max)
    
    color_sensor = profile.get_device().query_sensors()[1]
    # color_sensor.set_option(rs.option.enable_auto_exposure, False)
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)


    # Using preset HighAccuracy for recording
    if args.record_rosbag or args.record_imgs:
        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1.5  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    frame_count = 0
    try:
        with open(join(args.output_folder,'camera_pose.csv'), 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            while True:
                # Get frameset of color and depth
                frames = pipeline_D435.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                pose_frames = pipeline_T265.poll_for_frames()

                # Validate that both frames are valid as well as the pose frames. Maybe check tracker confidence to 3?
                if not aligned_depth_frame or not color_frame or not pose_frames:
                    continue
                
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                pose = pose_frames.get_pose_frame()
                data = pose.get_pose_data()

                if args.record_imgs:
                    if frame_count == 0:
                        writer.writerow(["Frame count", "Pos x", "Pos y", "Pos z", "Rot x", "Rot y", "Rot z", "Rot w", "Tracker confidence"])
                        save_intrinsic_as_json(
                            join(args.output_folder, "camera_intrinsic.json"),
                            color_frame)
                    cv2.imwrite("%s/%06d.png" % \
                            (path_depth, frame_count), depth_image)
                    cv2.imwrite("%s/%06d.jpg" % \
                            (path_color, frame_count), color_image)
                    print("Saved color + depth image %06d" % frame_count)
                    writer.writerow([frame_count, data.translation.x, data.translation.y, data.translation.z,
                                                  data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w, data.tracker_confidence])
                    frame_count += 1

                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                #depth image is 1 channel, color is 3 channels
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
                bg_removed = np.where((depth_image_3d > clipping_distance) | \
                        (depth_image_3d <= 0), grey_color, color_image)

                # Render images
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)
                images = np.hstack((bg_removed, depth_colormap))
                cv2.namedWindow('Recorder Realsense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('Recorder Realsense', images)
                cv2.imshow('Recorder Realsense', bg_removed)
                key = cv2.waitKey(1)

                # if 'esc' button pressed, escape loop and exit program
                if key == 27:
                    cv2.destroyAllWindows()
                    break
    finally:
        for pipeline in pipelines:
            pipeline.stop()
