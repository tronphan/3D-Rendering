import os
import open3d as o3d
import pandas as pd
from drawing import *
import quaternion

T265_EX =     [[ 0.999968402, -0.006753626, -0.004188075, -0.015890727],
               [-0.006685408, -0.999848172,  0.016093893,  0.028273059],
               [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
               [           0,            0,            0,            1]]

FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

def create_RGBD_point_cloud(file_number, config):
    # Create point cloud from RGB + Depth files
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=config["max_depth"],convert_rgb_to_intensity=False) # max_depth = 3
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def get_pose_T265(pose_data):
    trans = np.identity(4)
    trans[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion(
        [pose_data["Rot w"],pose_data["Rot x"],pose_data["Rot y"],pose_data["Rot z"]])
    trans[:3,3] = [pose_data["Pos x"],pose_data["Pos y"],pose_data["Pos z"]]
    trans = np.dot(trans, T265_EX)
    trans = np.dot(FLIP, trans)
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

def relative_trans2(pose_source, pose_target):
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

if __name__ == '__main__':
    print("##############################################")
    print("1. Load two point clouds and show initial pose")
    print("##############################################")
    source_file_number = 100
    target_file_number = 130
    max_depth = 1.0 # m
    path_dataset = "dataset/realsense/"
    path_intrinsic = "dataset/realsense/camera_intrinsic.json" # intrinsic parameter
    config = {"max_depth": max_depth, "path_dataset": path_dataset, "path_intrinsic":path_intrinsic}

    source = create_RGBD_point_cloud(source_file_number, config) # Point cloud are built by combining depth and RGB image knowing the intrinsic matrix (see deprojection fonction)
    target = create_RGBD_point_cloud(target_file_number, config)

    print("Draw initial alignement")
    draw_registration_result_original_color(source, target)
    draw_registration_result(source, target)

    print("\n")
    print("##############################################")
    print("2. Apply registration with pose data")
    print("##############################################")
    # Load pose data from intel T265
    pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))
    pose_s = pose_data.iloc[source_file_number]
    pose_t = pose_data.iloc[target_file_number]

    T265_pose_source = get_pose_T265(pose_s)
    T265_pose_target = get_pose_T265(pose_t)

    source.transform(T265_pose_source)
    target.transform(T265_pose_target)

    # source.transform(T265_EX)
    # target.transform(T265_EX)

    # rel_trans = relative_trans2(pose_s, pose_t)
    # rel_trans = np.dot(np.linalg.inv(T265_EX),np.dot(rel_trans,T265_EX))

    draw_registration_result_original_color(source, target)
    draw_registration_result(source, target)

    # draw_registration_result_original_color(source, target, rel_trans)
    # draw_registration_result(source, target, rel_trans)

    # source.transform(relative_trans(pose_s, pose_t))
    # Not perfect, suffer from drift.

    # print("\n")
    # print("##############################################")
    # print("3. Point to plane ICP registration")
    # print("##############################################")    
    # current_transformation = np.identity(4)
    # print("Point-to-plane ICP registration is applied on original point")
    # print("clouds to refine the alignment. Distance threshold 0.02.")
    # source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #     radius=0.1, max_nn=30))
    # result_icp = o3d.registration.registration_icp(
    #     source, target, 0.02, current_transformation,
    #     o3d.registration.TransformationEstimationPointToPlane())
    # print(result_icp)
    # draw_registration_result_original_color(source, target,
    #                                         result_icp.transformation)
    # draw_registration_result(source, target,result_icp.transformation)


    # print("\n")
    # print("##############################################")
    # print("4. Colored point cloud registration")
    # print("##############################################")
    # # This is implementation of following paper
    # # J. Park, Q.-Y. Zhou, V. Koltun,
    # # Colored Point Cloud Registration Revisited, ICCV 2017
    # voxel_radius = [0.04, 0.02, 0.01]
    # max_iter = [50, 30, 14]
    # current_transformation = rel_trans
    # for scale in range(len(voxel_radius)):
    #     iter = max_iter[scale]
    #     radius = voxel_radius[scale]
    #     print([iter, radius, scale])

    #     print("3-1. Downsample with a voxel size %.2f" % radius)
    #     source_down = source.voxel_down_sample(radius)
    #     target_down = target.voxel_down_sample(radius)

    #     print("3-2. Estimate normal.")
    #     source_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    #     target_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    #     print("3-3. Applying colored point cloud registration")
    #     result_icp = o3d.registration.registration_colored_icp(
    #         source_down, target_down, radius, current_transformation,
    #         o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                 relative_rmse=1e-6,
    #                                                 max_iteration=iter))
    #     current_transformation = result_icp.transformation
    #     print(result_icp)
    # draw_registration_result_original_color(source, target,
    #                                         result_icp.transformation)
    # draw_registration_result(source, target,result_icp.transformation)
    # print(result_icp.transformation)
    # print(relative_trans(pose_s, pose_t))


    # source.transform(result_icp.transformation)
    # draw_registration_result_original_color(source, target)
    # draw_registration_result(source, target)