# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/colored_pointcloud_registration.py

import numpy as np
import copy
import open3d as o3d
import json
from initialize_config import initialize_config
import pandas as pd
from drawing import draw_geometries_flip
import math as m

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_file,
        depth_file,
        depth_trunc=config["max_depth"],
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image

def create_RGBD_point_cloud(rgb_file, depth_file, config):
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_file, depth_file)
    rgbd_image = read_rgbd_image(rgb_file, depth_file, False, config)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def draw_geometries_flip(pcds, id):
    if id == 0:
        flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    else:
        flip_transform = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcds_transform = []
    for pcd in pcds:
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.transform(flip_transform)
        pcds_transform.append(pcd_temp)
    o3d.visualization.draw_geometries(pcds_transform)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def build_trans_from_pose(pose):
    q_x = pose["Rot x"]
    q_y = pose["Rot y"]
    q_z = pose["Rot z"]
    q_w = pose["Rot w"]
    t_x = pose["Pos x"]
    t_y = pose["Pos y"]
    t_z = pose["Pos z"]
    # rot = o3d.geometry.get_rotation_matrix_from_quaternion([q_x,q_y,q_z,q_w])
    # trans = np.identity(4)
    # trans[:3,:3] = rot
    # trans[:3,3] = [t_x,t_y,t_z]
    print("track_confidence: %d" % pose["Tracker confidence"])
    trans   = np.array([[q_w**2 + q_x**2 - q_y**2 - q_z**2,     2*(q_x*q_y - q_w*q_z),                  2*(q_w*q_y + q_x*q_z),                  t_x],
                        [2*(q_x*q_y + q_w*q_z),                 q_w**2 - q_x**2 + q_y**2 - q_z**2,      2*(q_y*q_z - q_w*q_x),                  t_y],
                        [2*(q_x*q_z - q_w*q_y),                 2*(q_w*q_x + q_y*q_z),                  q_w**2 - q_x**2 - q_y**2 + q_z**2,      t_z],
                        [0,                                     0,                                      0,                                      1]])
    return trans

def build_trans_from_pose2(pose_s, pose_t):
    q_x_s = pose_s["Rot x"]
    q_y_s = pose_s["Rot y"]
    q_z_s = pose_s["Rot z"]
    q_w_s = pose_s["Rot w"]
    t_x_s = pose_s["Pos x"]
    t_y_s = pose_s["Pos y"]
    t_z_s = pose_s["Pos z"]
    rot_s = o3d.geometry.get_rotation_matrix_from_quaternion([q_w_s,q_x_s,q_y_s,q_z_s])

    q_x_t = pose_t["Rot x"]
    q_y_t = pose_t["Rot y"]
    q_z_t = pose_t["Rot z"]
    q_w_t = pose_t["Rot w"]
    t_x_t = pose_t["Pos x"]
    t_y_t = pose_t["Pos y"]
    t_z_t = pose_t["Pos z"]
    rot_t = o3d.geometry.get_rotation_matrix_from_quaternion([q_w_t,q_x_t,q_y_t,q_z_t])
    rot = np.dot(rot_s, rot_t.transpose())

    trans = np.identity(4)
    trans[:3,:3] = rot
    trans[:3,3] = [t_x_s-t_x_t,t_y_s-t_y_t,t_z_s-t_z_t]
    print("track_confidence: %d" % pose_s["Tracker confidence"])
    return trans


if __name__ == "__main__":
    print("##############################################")
    print("1. Load two point clouds and show initial pose")
    print("##############################################")
    # source = o3d.io.read_point_cloud("../../TestData/ColoredICP/frag_115.ply")
    # target = o3d.io.read_point_cloud("../../TestData/ColoredICP/frag_116.ply")

    s = 500
    t = s+300
    with open("realsense.json") as json_file:
        config = json.load(json_file)
        initialize_config(config)
    source_color = o3d.io.read_image("color/%06d.jpg" % s)
    source_depth = o3d.io.read_image("depth/%06d.png" % s)
    target_color = o3d.io.read_image("color/%06d.jpg" % t)
    target_depth = o3d.io.read_image("depth/%06d.png" % t)
    source = create_RGBD_point_cloud(source_color,source_depth,config)
    target = create_RGBD_point_cloud(target_color,target_depth,config)

    # draw initial alignment
    current_transformation = np.identity(4)
    draw_registration_result_original_color(source, target, current_transformation)
    draw_registration_result(source, target,current_transformation)

    print("\n")
    print("##############################################")
    print("2. Apply registration with pose data")
    print("##############################################")
    # Use T265 pose data
    pose_data = pd.read_csv("camera_pose.csv")

    pose_s = pose_data.iloc[s]
    pose_t = pose_data.iloc[t]
    q_x_s = pose_s["Rot x"]
    q_y_s = pose_s["Rot y"]
    q_z_s = pose_s["Rot z"]
    q_w_s = pose_s["Rot w"]
    t_x_s = pose_s["Pos x"]
    t_y_s = pose_s["Pos y"]
    t_z_s = pose_s["Pos z"]
    rot_s = o3d.geometry.get_rotation_matrix_from_quaternion([q_w_s,q_x_s,q_y_s,q_z_s])
    trans_s = np.identity(4)
    trans_s[:3,:3] = rot_s

    q_x_t = pose_t["Rot x"]
    q_y_t = pose_t["Rot y"]
    q_z_t = pose_t["Rot z"]
    q_w_t = pose_t["Rot w"]
    t_x_t = pose_t["Pos x"]
    t_y_t = pose_t["Pos y"]
    t_z_t = pose_t["Pos z"]
    rot_t = o3d.geometry.get_rotation_matrix_from_quaternion([q_w_t,q_x_t,q_y_t,q_z_t])
    trans_t = np.identity(4)
    trans_t[:3,:3] = rot_t


    # trans_source = build_trans_from_pose(pose_data.iloc[s])
    # trans_target = build_trans_from_pose(pose_data.iloc[s])
    trans_rel = build_trans_from_pose2(pose_data.iloc[s],pose_data.iloc[t])
    print(trans_rel)
    trans_test = np.identity(4)
    trans_test[:3,3] = trans_rel[:3,3]
    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip)
    target.transform(flip)
    draw_registration_result_original_color(source.transform(trans_s), target.transform(trans_t), trans_test)
    draw_registration_result(source, target, trans_test)

    source.transform(trans_test)


    # trans_source = np.identity(4)
    # trans_target = np.identity(4)
    # rot_source = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # rot_target = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # trans_source[:3,:3] = rot_source
    # trans_target[:3,:3] = rot_target
    # trans_source[:3,3] = [0,2,0] #x y z
    # trans_target[:3,3] = [0,0,0]
    # print(trans_source)
    # print(trans_target)

    # pcds = [source.transform(trans_source), target.transform(trans_target)]
    # draw_geometries_flip(pcds,1)


    # # point to plane ICP
    # current_transformation = np.identity(4)
    # print("2. Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. Distance threshold 0.02.")
    # result_icp = o3d.registration.registration_icp(
    #     source, target, 0.02, current_transformation,
    #     o3d.registration.TransformationEstimationPointToPlane())
    # print(result_icp)
    # draw_registration_result_original_color(source, target,
    #                                         result_icp.transformation)




    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("\n")
    print("##############################################")
    print("3. Colored point cloud registration")
    print("##############################################")
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


    # print("Test")
    # A = np.dot(np.linalg.inv(trans_rel),result_icp.transformation)
    # print(A)
    # print(trans_rel)
    # print(result_icp.transformation)
    # res = np.dot(trans_rel,A)
    # print(res)
    # draw_registration_result_original_color(source, target, res)

