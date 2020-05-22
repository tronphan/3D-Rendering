import open3d as o3d
import pandas as pd
import numpy as np
import quaternion
import os
import open3d as o3d
import sys
sys.path.append("../Utility")
from file import join
# from drawing import *

T265_EX =     [[ 0.999968402, -0.006753626, -0.004188075, -0.015890727],
               [-0.006685408, -0.999848172,  0.016093893,  0.028273059],
               [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
               [           0,            0,            0,            1]]
FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

def create_RGBD_image(file_number, config):
    # Create point cloud from RGB + Depth files
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=1.0,convert_rgb_to_intensity=False) # max_depth = 3
    return rgbd_image

def create_RGBD_point_cloud(file_number, config):
    rgbd_image = create_RGBD_image(file_number, config)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def init_posegraph(file_number, pose_data):
    #Create a posegraph and initialize the node with the T265 pose data
    posegraph = o3d.registration.PoseGraph()
    odometry = get_pose_T265(pose_data.iloc[0]) #OK
    posegraph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry))) #OK

    for i in range(1, len(pose_data.index)):
        node = get_pose_T265(pose_data.iloc[i]) #OK 
        posegraph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(node))) #Ok

        [success, trans, info] = register_one_rgbd_pair(i-1, pose_data.iloc[i-1], i, pose_data.iloc[i], config)
        # trans_odometry = np.dot(trans, trans_odometry)
        # trans_odometry_inv = np.linalg.inv(trans_odometry)
        # posegraph.nodes.append(o3d.registration.PoseGraphNode(trans_odometry_inv))
        edge = o3d.registration.PoseGraphEdge(i-1, i, trans, info, uncertain=True)
        posegraph.edges.append(edge)
        print(f"Register frame {i}")
    return posegraph

def register_one_rgbd_pair(s, pose_s, t, pose_t, config):
    source = create_RGBD_point_cloud(s, config)
    target = create_RGBD_point_cloud(t, config)
    # trans_s = get_pose_T265(pose_s)
    # trans_t = get_pose_T265(pose_t)

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = relative_trans(pose_s, pose_t)
    for scale in range(len(voxel_radius)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
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
    trans = result_icp.transformation
    
    # rel_trans = relative_trans(pose_s, pose_t)
    # trans = np.dot(np.linalg.inv(T265_EX),np.dot(rel_trans,T265_EX))
    info = o3d.registration.get_information_matrix_from_point_clouds(source, target, radius, trans)
    # info = np.zeros((6,6))
    success = True
    return [success, trans, info]

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
    translation = np.dot(quaternion.as_rotation_matrix(r), np.subtract(translation_source, translation_target))
    rel_trans[:3,3] = translation
    # trans_source = get_pose_T265(pose_source)
    # trans_target = get_pose_T265(pose_target)
    # rel_trans = np.dot(np.linalg.inv(trans_source), trans_target)
    return rel_trans
    
def run_posegraph_optimization(pose_graph_name, pose_graph_optimized_name,
                               max_correspondence_distance,
                               preference_loop_closure):
    # to display messages from o3d.registration.global_optimization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    method = o3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance,
        edge_prune_threshold=0.25,
        preference_loop_closure=preference_loop_closure,
        reference_node=0)
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    o3d.registration.global_optimization(pose_graph, method, criteria, option)
    o3d.io.write_pose_graph(pose_graph_optimized_name, pose_graph)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

if __name__ == '__main__':
    # First attemp with only 10 images
    file_number = range(350,362,1)

    max_depth = 1.5 # m
    path_dataset = "dataset/realsense/"
    path_intrinsic = "dataset/realsense/camera_intrinsic.json" # intrinsic parameter
    config = {"max_depth": max_depth, "path_dataset": path_dataset, "path_intrinsic":path_intrinsic, "max_depth_diff":0.07, "preference_loop_closure_odometry": 0.1,
    "preference_loop_closure_registration": 5.0}
    pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))

    posegraph = init_posegraph(file_number, pose_data.iloc[file_number])
    # o3d.io.write_pose_graph("posegraph.json", posegraph)
    # run_posegraph_optimization("posegraph.json","posegraph_opti.json",
    #         max_correspondence_distance = 1.4*0.05,
    #         preference_loop_closure = \
    #         config["preference_loop_closure_registration"])
    # posegraph = o3d.io.read_pose_graph("posegraph_opti.json")

    # volume = o3d.integration.ScalableTSDFVolume(
    #     voxel_length=1.5 / 512.0,
    #     sdf_trunc=0.04,
    #     color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    # for i in range(len(posegraph.nodes)):
    #     rgbd = create_RGBD_image(file_number[i], config)
    #     pose = posegraph.nodes[i].pose
    #     # volume.integrate(rgbd, o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]), np.linalg.inv(pose))
    #     volume.integrate(rgbd, o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]), pose)
    #     print(f"Integrating frame {i}/{len(file_number)-1}")
    # mesh = volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()
    # final_pcd = o3d.geometry.PointCloud()
    # final_pcd.points = mesh.vertices
    # # final_pcd.colors = mesh.vertex_colors
    # # final_pcd.normals = mesh.normals
    # pcd = o3d.geometry.PointCloud()
    # # pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # pcd.points = mesh.vertices
    # pcd.colors = mesh.vertex_colors
    # # pcd.estimate_normals()
    # o3d.visualization.draw_geometries([pcd])



    pcd_combined = o3d.geometry.PointCloud()
    for i in range(len(posegraph.nodes)):
        pcd = create_RGBD_point_cloud(file_number[i], config)
        pcd.transform(np.linalg.inv(posegraph.nodes[i].pose))
        pcd_combined += pcd
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.001)
    o3d.visualization.draw_geometries([pcd_combined_down])
    