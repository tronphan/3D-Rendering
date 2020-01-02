# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/rgbd_odometry.py

import open3d as o3d
import numpy as np
import pandas as pd
import time

def calc_transform(s_pose, t_pose):
    s_track_confidence = s_pose["Tracker confidence"]
    t_track_confidence = t_pose["Tracker confidence"]

    if s_track_confidence >= 2 and t_track_confidence >= 2:
        success = True
        q_x = t_pose["Rot x"] - s_pose["Rot x"]
        q_y = t_pose["Rot y"] - s_pose["Rot y"]
        q_z = t_pose["Rot z"] - s_pose["Rot z"]
        q_w = t_pose["Rot w"] - s_pose["Rot w"]

        t_x = t_pose["Pos x"] - s_pose["Pos x"]
        t_y = t_pose["Pos y"] - s_pose["Pos y"]
        t_z = t_pose["Pos z"] - s_pose["Pos z"]
        
        trans   = np.array([    [(1 - 2 * q_y*q_y - 2 * q_z*q_z),    2 * q_x*q_y - 2 * q_z*q_w,      (2 * q_x*q_z + 2 * q_y*q_w),       t_x],
                                [(2 * q_x*q_y + 2 * q_z*q_w),        1 - 2 * q_x*q_x - 2 * q_z*q_z,  (2 * q_y*q_z - 2 * q_x*q_w),       t_y],
                                [(2 * q_x*q_z - 2 * q_y*q_w),        2 * q_y*q_z + 2 * q_x*q_w,      (1 - 2 * q_x*q_x - 2 * q_y*q_y),   t_z],
                                [0,                                  0,                              0,                                  1  ]  ])
        info = np.identity(6)
        return [success, trans, info]

    else:
        return [False, np.identity(4), np.identity(6)]

if __name__ == "__main__":    
    # pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
    #     "../../TestData/camera_primesense.json")
    # print(pinhole_camera_intrinsic.intrinsic_matrix)

    # source_color = o3d.io.read_image("../../TestData/RGBD/color/00000.jpg")
    # source_depth = o3d.io.read_image("../../TestData/RGBD/depth/00000.png")
    # target_color = o3d.io.read_image("../../TestData/RGBD/color/00001.jpg")
    # target_depth = o3d.io.read_image("../../TestData/RGBD/depth/00001.png")

    pose_data = pd.read_csv("dataset/realsense/camera_pose.csv")
    pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
        "dataset/realsense/camera_intrinsic.json")
    print(pinhole_camera_intrinsic.intrinsic_matrix)
    s = 44
    t = 54
    source_color = o3d.io.read_image("dataset/realsense/color/000044.jpg")
    source_depth = o3d.io.read_image("dataset/realsense/depth/000044.png")
    target_color = o3d.io.read_image("dataset/realsense/color/000054.jpg")
    target_depth = o3d.io.read_image("dataset/realsense/depth/000054.png")


    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, pinhole_camera_intrinsic)

    option = o3d.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)
    print("\n")

    t0 = time.time()
    [success_color_term, trans_color_term,
     info] = o3d.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
         odo_init, o3d.odometry.RGBDOdometryJacobianFromColorTerm(), option)

    t1 = time.time()
    [success_hybrid_term, trans_hybrid_term,
     info] = o3d.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
         odo_init, o3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    t2 = time.time()
    [success_T265, trans_T265,
     info] = calc_transform(pose_data.iloc[s], pose_data.iloc[t])
    t3 = time.time()

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        print(t1-t0)
        # source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     source_rgbd_image, pinhole_camera_intrinsic)
        # source_pcd_color_term.transform(trans_color_term)
        # o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term])
    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        print(t2-t1)
        # source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
        #     source_rgbd_image, pinhole_camera_intrinsic)
        # source_pcd_hybrid_term.transform(trans_hybrid_term)
        # o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
    if success_T265:
        print("Using T265 Odometry")
        print(trans_T265)
        print(t3-t2)
