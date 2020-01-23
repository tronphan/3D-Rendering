# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/colored_pointcloud_registration.py

import os
import sys
sys.path.append("config")
from initialize_config import initialize_config
import json
import numpy as np
import pandas as pd
import open3d as o3d
from drawing import *

def create_RGBD_point_cloud(file_number, config):
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=config["max_depth"],convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def trans_from_pose(pose_data):
    rot = o3d.geometry.get_rotation_matrix_from_quaternion(
        [pose_data["Rot w"],pose_data["Rot x"],pose_data["Rot y"],pose_data["Rot z"]])
    trans = np.identity(4)
    trans[:3,:3] = rot
    trans[:3,3] = [pose_data["Pos x"],pose_data["Pos y"],pose_data["Pos z"]]
    return trans
    
if __name__ == "__main__":
    print("##############################################")
    print("1. Load two point clouds and show initial pose")
    print("##############################################")
    # Load RGBD file from intel D435
    source_file_number = 1912
    target_file_number = 1922
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
    draw_registration_result_original_color(source, target, current_transformation)
    draw_registration_result(source, target, current_transformation)


    print("\n")
    print("##############################################")
    print("2. Apply registration with pose data")
    print("##############################################")
    # Load pose data from intel T265
    pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))

    pose_s = pose_data.iloc[source_file_number]
    pose_t = pose_data.iloc[target_file_number]

    trans_source = trans_from_pose(pose_s)
    trans_target = trans_from_pose(pose_t)
    print(trans_source)
    print(trans_target)

    draw_registration_result_original_color(source.transform(trans_source), target.transform(trans_target), np.identity(4))
    draw_registration_result(source, target, np.identity(4))


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
