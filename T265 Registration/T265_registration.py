# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# See original file in Open3D github: examples/Python/Advanced/colored_pointcloud_registration.py

# This file shows the capabilities to have a first rough global alignement by using the pose data
# obtained by an additional camera Intel T265 that is physically attached on top of a RGBD camera Intel D435.

import os
import sys
sys.path.append("config")
from initialize_config import initialize_config
import json
import numpy as np
import pandas as pd
import open3d as o3d
from drawing import *
import quaternion
import time

def create_RGBD_point_cloud(file_number, config):
    # Create point cloud from RGB + Depth files
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=config["max_depth"],convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def trans_from_pose(pose_data):
    # Compute the transformation matrix from the T265 pose data: rotration + translation
    # Check: https://github.com/IntelRealSense/librealsense/blob/master/doc/t265.md
    rot = o3d.geometry.get_rotation_matrix_from_quaternion(
        [pose_data["Rot w"],pose_data["Rot x"],pose_data["Rot y"],pose_data["Rot z"]])
    trans = np.identity(4)
    trans[:3,:3] = rot
    trans[:3,3] = [pose_data["Pos x"],pose_data["Pos y"],pose_data["Pos z"]]
    return trans

def relative_trans(pose_source, pose_target):
    # Compute the relative transformation matrix by using the pose data of the source and target
    translation_source = np.array([pose_source["Pos x"],pose_source["Pos y"],pose_source["Pos z"]])
    translation_target = np.array([pose_target["Pos x"],pose_target["Pos y"],pose_target["Pos z"]])
    q = np.quaternion(pose_source["Rot w"],pose_source["Rot x"],pose_source["Rot y"],pose_source["Rot z"])
    r = np.quaternion(pose_target["Rot w"],-pose_target["Rot x"],-pose_target["Rot y"],-pose_target["Rot z"])
    p = r*q
    rot = quaternion.as_rotation_matrix(p)
    rel_trans = np.identity(4)
    rel_trans[:3,:3] = rot
    translation = np.dot(quaternion.as_rotation_matrix(r),np.subtract(translation_source, translation_target))
    rel_trans[:3,3] = translation
    return rel_trans
    
if __name__ == "__main__":
    print("##############################################")
    print("1. Load two point clouds and show initial pose")
    print("##############################################")
    # Load RGBD file from intel D435
    source_file_number = 600
    target_file_number = 700
    with open("config/realsense.json") as json_file:
        config = json.load(json_file)
        initialize_config(config)
    source = create_RGBD_point_cloud(source_file_number,config)
    target = create_RGBD_point_cloud(target_file_number,config)

    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip)
    target.transform(flip)

    print("Draw initial alignement")
    current_transformation = np.identity(4)
    draw_registration_result_original_color(target, source, current_transformation)
    draw_registration_result(source, target, current_transformation)


    print("\n")
    print("##############################################")
    print("2. Apply registration with pose data")
    print("##############################################")
    # Load pose data from intel T265
    pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))

    pose_s = pose_data.iloc[source_file_number]
    pose_t = pose_data.iloc[target_file_number]

    start1 = time.time()
    trans_source = trans_from_pose(pose_s)
    trans_target = trans_from_pose(pose_t)
    rel_trans1 = np.dot(np.linalg.inv(trans_target),trans_source)
    stop1 = time.time()
    duration1 = stop1-start1

    start2 = time.time()
    rel_trans2 = relative_trans(pose_s, pose_t)
    stop2 = time.time()
    duration2 = stop2-start2

    print(duration1)
    print(duration2)
    print(rel_trans1)
    print(rel_trans2)

    # draw_registration_result_original_color(source.transform(trans_source), target.transform(trans_target), np.identity(4))
    # draw_registration_result(source, target, np.identity(4))

    draw_registration_result_original_color(source, target, rel_trans2)
    draw_registration_result(source, target, rel_trans2)
    source.transform(rel_trans2)

    
    print("\n")
    print("##############################################")
    print("3. Colored point cloud registration")
    print("##############################################")
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    print(result_icp.transformation)
    draw_registration_result_original_color(source, target,
                                            result_icp.transformation)
    draw_registration_result(source, target,result_icp.transformation)