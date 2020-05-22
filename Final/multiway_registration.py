# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/multiway_registration.py
import os
import sys
sys.path.append("config")
import open3d as o3d
import numpy as np
from file import *
import json
from initialize_config import initialize_config
voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

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

def load_point_clouds(voxel_size=0.0):
    with open("config/realsense.json") as json_file:
        config = json.load(json_file)
        initialize_config(config)
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    # pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))
    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    start = 1200
    stop = start+30

    pcds = []
    for i in range(350, 370):
        rgbd = read_rgbd_image(color_files[i], depth_files[i], config)
        pcd = create_RGBD_point_cloud(rgbd, config)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds

def pairwise_registration(source, target):
    source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * 2, max_nn=30))
    icp_coarse = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("source %d :: target %d" %(source_id, target_id))
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


if __name__ == "__main__":
    pcds_down = load_point_clouds(voxel_size)
    o3d.visualization.draw_geometries(pcds_down)

    print("Full registration ...")
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.registration.global_optimization(
        pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.registration.GlobalOptimizationConvergenceCriteria(), option)

    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    o3d.visualization.draw_geometries(pcds_down)

    print("Make a combined point cloud")
    pcds = load_point_clouds(voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.002)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down])
