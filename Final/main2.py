import os
import sys
sys.path.append("config")
import json
from file import *
from initialize_config import initialize_config
import open3d as o3d
import pandas as pd
import numpy as np
import quaternion
import cv2

def read_rgbd_image(color_file, depth_file, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_trunc=config["max_depth"],
        convert_rgb_to_intensity=False)
    return rgbd_image

def create_RGBD_point_cloud(rgbd_image, config):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def get_pose_matrix(pose):
    # Compute the transformation matrix
    q = np.quaternion(pose["Rot w"], pose["Rot x"], pose["Rot y"], pose["Rot z"])
    rotation = quaternion.as_rotation_matrix(q)
    translation = np.array([pose["Pos x"], pose["Pos y"], pose["Pos z"]])
    pose_matrix = np.identity(4)
    pose_matrix[:3,:3] = rotation
    pose_matrix[:3,3] = translation
    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]] 
    return np.dot(pose_matrix, flip)

if __name__ == "__main__":
    print("##############################################")
    print("1. Load data")
    print("##############################################")
    with open("config/realsense.json") as json_file:
        config = json.load(json_file)
        initialize_config(config)
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    pose_data_raw = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))
    [color_files_raw, depth_files_raw] = get_rgbd_file_lists(config["path_dataset"])

    print("\n")
    print("##############################################")
    print("3. Make a combined point cloud")
    print("##############################################")

    # Remove blurry images
    color_files = []
    depth_files = []
    pose_data_index = []
    threshold = 50
    print("Check blurry images")
    for i in range(len(color_files_raw)):
        image = cv2.imread(color_files_raw[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm >= threshold:
            color_files.append(color_files_raw[i])
            depth_files.append(depth_files_raw[i])
            pose_data_index.append(i)
    print("Done")

    pose_data = pose_data_raw.iloc[pose_data_index]

    rgbd = read_rgbd_image(color_files[0], depth_files[0], config)
    pcd = create_RGBD_point_cloud(rgbd, config)
    pcd.transform(get_pose_matrix(pose_data.iloc[0]))
    pcd_combined = pcd
    pcd_prev = pcd

    # voxel_radius = [0.04, 0.02, 0.01]
    # max_iter = [50, 30, 14]
    voxel_radius = [0.01]
    max_iter = [4]

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=2.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    volume.integrate(rgbd, intrinsic, np.linalg.inv(get_pose_matrix(pose_data.iloc[0])))

    for i in range(len(color_files)):
        print("File %d/%d combined" %(i, len(color_files)))
        rgbd = read_rgbd_image(color_files[i], depth_files[i], config)
        pcd = create_RGBD_point_cloud(rgbd, config)
        current_transformation = get_pose_matrix(pose_data.iloc[i])
        for scale in range(len(voxel_radius)):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            source_down = pcd.voxel_down_sample(radius)
            target_down = pcd_combined.voxel_down_sample(radius)
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            result_icp = o3d.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter))
            current_transformation = result_icp.transformation
        pcd_prev = pcd
        pcd_combined += pcd.transform(current_transformation)
        pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.003)
        volume.integrate(
            rgbd, intrinsic,
            np.linalg.inv(current_transformation))

    # pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.003)
    # o3d.visualization.draw_geometries([pcd_combined])

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = mesh.vertices
    final_pcd.colors = mesh.vertex_colors
    o3d.visualization.draw_geometries([final_pcd])