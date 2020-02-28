import numpy as np
import quaternion
import pandas as pd
import json
import open3d as o3d

import os, sys
sys.path.append("config")
from initialize_config import initialize_config
from registration import create_RGBD_point_cloud

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

def load_point_clouds(pcds, voxel_size=0.0):
    for pcd in pcds:
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds_down

def create_RGBD(file_number, config):
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=config["max_depth"],convert_rgb_to_intensity=False)
    return rgbd_image

def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.registration.PoseGraph()
    trans_odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(s - sid,
                                                   t - sid,
                                                   trans,
                                                   info,
                                                   uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 \
                    and t % config['n_keyframes_per_n_frame'] == 0:
                print(
                    "Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                    % (fragment_id, n_fragments - 1, s, t))
                [success, trans,
                 info] = register_one_rgbd_pair(s, t, color_files, depth_files,
                                                intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(
                        o3d.registration.PoseGraphEdge(s - sid,
                                                       t - sid,
                                                       trans,
                                                       info,
                                                       uncertain=True))
    o3d.io.write_pose_graph(
        join(path_dataset, config["template_fragment_posegraph"] % fragment_id),
        pose_graph)

if __name__ == "__main__":
    print("##############################################")
    print("1. Load point clouds")
    print("##############################################")
    # Load RGBD file from intel D435
    pcds_file_number = range(160,190)
    with open("config/realsense.json") as json_file:
        config = json.load(json_file)
        initialize_config(config)
    pcds = [create_RGBD_point_cloud(file_number, config) for file_number in pcds_file_number] 

    print("Draw initial alignement")
    o3d.visualization.draw_geometries(pcds)

    print("\n")
    print("##############################################")
    print("2. Apply registration with pose data")
    print("##############################################")
    # Load pose data from intel T265
    pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))
    pcds = [pcds[i].transform(get_pose_matrix(pose_data.iloc[pcds_file_number[i]])) for i in range(len(pcds))]
    o3d.visualization.draw_geometries(pcds)

    print("\n")
    print("##############################################")
    print("3. Affine Registration")
    print("##############################################")

    make_posegraph_for_fragment(path_dataset, sid, eid, color_files, depth_files, fragment_id, n_fragments, intrinsic, with_opencv, config)

    # print("\n")
    # print("##############################################")
    # print("3. Make a combined point cloud")
    # print("##############################################")
    # # pcds_down = load_point_clouds(pcds, voxel_size = 0.02)
    # pcd_combined = o3d.geometry.PointCloud()
    # print("Begin")
    # for point_id in range(len(pcds)):
    #     # pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    #     pcd_combined += pcds[point_id]
    # pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.003)
    # # o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
    # print("Done")
    # o3d.visualization.draw_geometries([pcd_combined_down])


    print("\n")
    print("##############################################")
    print("4. Integrate several point cloud")
    print("##############################################")
    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=0.002, #2.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)

    for file_number in pcds_file_number:
        rgbd = create_RGBD(file_number, config)
        volume.integrate(
            rgbd, o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]),
             np.linalg.inv(get_pose_matrix(pose_data.iloc[file_number])))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    o3d.visualization.draw_geometries([pcd])
