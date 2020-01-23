# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/icp_registration.py

import json
import open3d as o3d
import numpy as np
import quaternion
import copy
from initialize_config import initialize_config
from drawing import draw_geometries_flip
import pandas as pd
from colored_pointcloud_registration import draw_registration_result_original_color

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

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

def compute_odometry(source_pose, target_pose):
    s_track_confidence = source_pose["Tracker confidence"]
    t_track_confidence = target_pose["Tracker confidence"]

    if s_track_confidence == 3 and t_track_confidence == 3:
        quatA = np.quaternion(source_pose["Rot w"],source_pose["Rot x"],source_pose["Rot y"],source_pose["Rot z"])
        quatB = np.quaternion(target_pose["Rot w"],target_pose["Rot x"],target_pose["Rot y"],target_pose["Rot z"])
        quatC = quatA*quatB.inverse()
        trans = quaternion.as_rotation_matrix(quatC)

        t_x = target_pose["Pos x"] - source_pose["Pos x"]
        t_y = target_pose["Pos y"] - source_pose["Pos y"]
        t_z = target_pose["Pos z"] - source_pose["Pos z"]
        translation = np.array([[t_x, t_y, t_z]])

        trans = np.concatenate((trans,translation.T), axis=1)
        trans = np.concatenate((trans,np.array([[0,0,0,1]])), axis=0)
        return [True, trans, np.identity(6)]

    else:
        return [False, np.identity(4), np.identity(6)]

if __name__ == "__main__":
    s = 429
    t = 490
    pose_data = pd.read_csv("camera_pose.csv")

    source_color = o3d.io.read_image("color/%06d.jpg" % s)
    source_depth = o3d.io.read_image("depth/%06d.png" % s)

    target_color = o3d.io.read_image("color/%06d.jpg" % t)
    target_depth = o3d.io.read_image("depth/%06d.png" % t)

    with open("realsense.json") as json_file:
            config = json.load(json_file)
            initialize_config(config)

    pcd_source = create_RGBD_point_cloud(source_color,source_depth,config)
    pcd_target = create_RGBD_point_cloud(target_color,target_depth,config)

    [succes, trans_init, info] = compute_odometry(pose_data.iloc[t], pose_data.iloc[s])
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                          [-0.139, 0.967, -0.215, 0.7],
    #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

    # print("Source PointCloud image %d" % s)
    # draw_geometries_flip([pcd_source])

    # print("Target PointCloud %d" % t)
    # draw_geometries_flip([pcd_target])

    if succes:
        print("Initial alignment")
        # draw_registration_result(pcd_source, pcd_target, trans_init)
        draw_registration_result_original_color(pcd_source, pcd_target, np.identity(4))


    # source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")

    # threshold = 0.02
    # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
    #                          [-0.139, 0.967, -0.215, 0.7],
    #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # draw_registration_result(source, target, trans_init)
    # print("Initial alignment")
    # evaluation = o3d.registration.evaluate_registration(source, target,
    #                                                     threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p.transformation)

    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # # draw_registration_result(source, target, reg_p2l.transformation)
    # draw_registration_result(source, target, np.identity(4))
