# examples/Python/Advanced/multiway_registration.py
import os
import open3d as o3d
import numpy as np
import quaternion
import pandas as pd

# add multiprocessing
# avoid all pairwise_registration

voxel_size = 0.005
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5

T265_EX =     [[ 0.999968402, -0.006753626, -0.004188075, -0.015890727],
               [-0.006685408, -0.999848172,  0.016093893,  0.028273059],
               [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
               [           0,            0,            0,            1]]

T265_EX_INV = np.linalg.inv(T265_EX)

FLIP = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

file_number = range(200,400,7)
# file_number = range(350,362,1)
max_depth = 1.1 # or 1.4m

def create_RGBD_image(file_number, config):
    # Create point cloud from RGB + Depth files
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=max_depth,convert_rgb_to_intensity=False) # max_depth = 3
    return rgbd_image

def create_RGBD_point_cloud(file_number, config):
    rgbd_image = create_RGBD_image(file_number, config)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def load_point_clouds(voxel_size=0.0):
    # file_number = range(350,362,2)
    path_intrinsic = "dataset/realsense/camera_intrinsic.json" # intrinsic parameter
    config = {"path_dataset":"dataset/realsense/", "path_intrinsic":path_intrinsic}
    pcds = []
    for i in file_number:
        pcd = create_RGBD_point_cloud(i, config)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcds.append(pcd_down)
    return pcds

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

def pairwise_registration(source, target, pose_source, pose_target):
    rel_trans = relative_trans(pose_source, pose_target)
    rel_trans = np.dot(T265_EX_INV,np.dot(rel_trans,T265_EX))

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = rel_trans
    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        0.04, rel_trans)
    if evaluation.fitness < 0.4:
        return False, None, None 
    else:
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
        transformation_icp = result_icp.transformation
        information_icp = o3d.registration.get_information_matrix_from_point_clouds(
            source, target, radius, transformation_icp)
        return True, transformation_icp, information_icp 


def get_pose_T265(pose_data):
    trans = np.identity(4)
    trans[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion(
        [pose_data["Rot w"],pose_data["Rot x"],pose_data["Rot y"],pose_data["Rot z"]])
    trans[:3,3] = [pose_data["Pos x"],pose_data["Pos y"],pose_data["Pos z"]]
    trans = np.dot(trans, T265_EX)
    # trans = np.dot(FLIP, trans)
    return trans

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    pose = pd.read_csv("dataset/realsense/camera_pose.csv")
    # odometry = np.identity(4)
    # pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    odometry = get_pose_T265(pose.iloc[file_number[0]])
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
    n_pcds = len(pcds)
    # file_number = range(350,362,2)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print(f"Apply point-to-plane ICP frame : {source_id} and {target_id}")
            success, transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], pose.iloc[file_number[source_id]], pose.iloc[file_number[target_id]])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                # odometry = np.dot(transformation_icp, odometry)
                # pose_graph.nodes.append(
                #     o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                odometry = get_pose_T265(pose.iloc[file_number[source_id]])
                pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))

                if success:
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(source_id,
                                                       target_id,
                                                       transformation_icp,
                                                       information_icp,
                                                       uncertain=False))
            else:  # loop closure case
                if success:
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(source_id,
                                                       target_id,
                                                       transformation_icp,
                                                       information_icp,
                                                       uncertain=True))
    return pose_graph


if __name__ == "__main__":
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcds_down = load_point_clouds(voxel_size)
    # o3d.visualization.draw_geometries(pcds_down)

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
    # o3d.visualization.draw_geometries(pcds_down)

    # print("Make a combined point cloud")
    pcds = load_point_clouds(voxel_size)
    # pcd_combined = o3d.geometry.PointCloud()
    # for point_id in range(len(pcds)):
    #     pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    #     pcd_combined += pcds[point_id]
    # # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    # # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    # o3d.visualization.draw_geometries([pcd_combined])

    print("Integrate point clouds")
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=1.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    path_intrinsic = "dataset/realsense/camera_intrinsic.json" # intrinsic parameter
    config = {"path_dataset":"dataset/realsense/", "path_intrinsic":path_intrinsic}
    for i in range(len(pcds)):
        rgbd = create_RGBD_image(file_number[i], config)
        pose = pose_graph.nodes[i].pose
        volume.integrate(rgbd, o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]), np.linalg.inv(pose))
        print(f"Integrating frame {i}/{len(file_number)-1}")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = mesh.vertices
    final_pcd.colors = mesh.vertex_colors
    o3d.visualization.draw_geometries([final_pcd])