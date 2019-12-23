#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.

#####################################################
##           librealsense T265 example             ##
#####################################################

# First import the library
import pyrealsense2 as rs

import argparse
from os import makedirs
from os.path import exists, join
import shutil
import csv

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

if __name__ == "__main__":
    # Check input arguments
    parser = argparse.ArgumentParser(description="T265 Recorder. Please select one of the optional arguments")
    
    parser.add_argument("--output_folder",
                        default='dataset/realsense/',
                        help="set output folder")
                    
    parser.add_argument("--record_imgs",
                        default='store_true',
                        action='store_true',
                        help="Recording save color and depth images into realsense folder")

    args = parser.parse_args()

    path_output = args.output_folder
    path_pose = join(args.output_folder, "pose")

    make_clean_folder(path_output)
    make_clean_folder(path_pose)

    # Declare and create a RealSense pipeline, encapsulating the actual device and sensors
    pipe = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of fish eye cameras and request pose data. Only the pose data are streamed to the PC. The fish eye camera are used however in
    # the SLAM algorithm that runs on the T265 VPU.
    cfg = rs.config()

    if args.record_imgs:
        cfg.enable_stream(rs.stream.pose) #maybe add some config here?

    # Start streaming with requested config
    pipe.start(cfg)

    # Streaming loop
    frame_count = 0
    try:
        while True:
            # Wait for the next set of frames from the camera
            # frames = pipe.wait_for_frames()
            frames = pipe.poll_for_frames()

            if not frames:
                continue

            # Fetch pose frame
            pose = frames.get_pose_frame()

            # Maybe check pose data and discard if too close to each other.
            if pose:
                if args.record_imgs:
                    # Write in CSV file the pose data and the frame number.
                    data = pose.get_pose_data()
                    if frame_count == 0:
                        with open(join(path_pose,'camera_pose.csv'), 'w', newline='') as file:
                            writer = csv.writer(file, delimiter=',')
                            writer.writerow(["Timestamp", "Position", "Tracker confidence"])
                            writer.writerow([str(pose.timestamp), str(data.translation), str(data.tracker_confidence)])

                    # Store pose.timestamp, data.translation, data.rotation, data.mapper_confidence, data.tracker_confidence

                    # Print some of the pose data to the terminal
                    # print("Frame #{}".format(pose.frame_number))
                    # print("Position: {}".format(data.translation))

                    frame_count += 1

            if frame_count == 50:
                break

    finally:
        pipe.stop()