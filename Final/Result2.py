import os
import open3d as o3d
import pandas as pd
from drawing import *

T265_EX =     [[ 0.999968402, -0.006753626, -0.004188075, -0.015890727],
               [-0.006685408, -0.999848172,  0.016093893,  0.028273059],
               [-0.004296131, -0.016065384, -0.999861654, -0.009375589],
               [           0,            0,            0,            1]]

def create_RGBD_image(file_number, config):
    # Create point cloud from RGB + Depth files
    color = o3d.io.read_image(os.path.join(config["path_dataset"], "color/%06d.jpg" % file_number))
    depth = o3d.io.read_image(os.path.join(config["path_dataset"], "depth/%06d.png" % file_number))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,depth,depth_trunc=config["max_depth"],convert_rgb_to_intensity=False) # max_depth = 3
    return rgbd_image

def create_RGBD_point_cloud(file_number, config):
    rgbd_image = create_RGBD_image(file_number, config)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
        o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]))
    return pcd

def get_pose_T265(pose_data):
    trans = np.identity(4)
    trans[:3,:3] = o3d.geometry.get_rotation_matrix_from_quaternion(
        [pose_data["Rot w"],pose_data["Rot x"],pose_data["Rot y"],pose_data["Rot z"]])
    trans[:3,3] = [pose_data["Pos x"],pose_data["Pos y"],pose_data["Pos z"]]
    trans = np.dot(trans,T265_EX)
    return trans

if __name__ == '__main__':
    ls = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130]

    max_depth = 1.5 # m
    path_dataset = "dataset/realsense/"
    path_intrinsic = "dataset/realsense/camera_intrinsic.json" # intrinsic parameter
    config = {"max_depth": max_depth, "path_dataset": path_dataset, "path_intrinsic":path_intrinsic}
    pose_data = pd.read_csv(os.path.join(config["path_dataset"], "camera_pose.csv"))

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)

    registrations = []
    trans_odometry = np.identity(4)
    for i in range(100,120): # can be paralellised
        source_file_number = i
        target_file_number = i+1

        source = create_RGBD_point_cloud(source_file_number, config) # Point cloud are built by combining depth and RGB image knowing the intrinsic matrix (see deprojection fonction)
        target = create_RGBD_point_cloud(target_file_number, config)
    
        pose_s = pose_data.iloc[source_file_number]
        pose_t = pose_data.iloc[target_file_number]

        # T265_pose_source = get_pose_T265(pose_s)
        # T265_pose_target = get_pose_T265(pose_t)

        # source.transform(T265_pose_source)
        # target.transform(T265_pose_target)

        for scale in range(len(voxel_radius)):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            # print([iter, radius, scale])

            # print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            # print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            # print("3-3. Applying colored point cloud registration")
            result_icp = o3d.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                        relative_rmse=1e-6,
                                                        max_iteration=iter))
            current_transformation = result_icp.transformation
            # print(result_icp)
        # trans = np.dot(np.linalg.inv(T265_pose_target), np.dot(result_icp.transformation, T265_pose_source))
        trans_odometry = np.dot(result_icp.transformation,trans_odometry)
        registrations.append({"source":source_file_number, "trans": trans_odometry})
        print(f"{source_file_number} registered to {target_file_number}")

    volume = o3d.integration.ScalableTSDFVolume(
        voxel_length=  1.5 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(registrations)):
        rgbd = create_RGBD_image(registrations[i]['source'], config)
        volume.integrate(rgbd, o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"]), registrations[i]['trans'])
        print(f"Integrating frame {i}/{len(registrations)}")
        # if i == 2: break

    mesh = volume.extract_triangle_mesh()
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = mesh.vertices
    final_pcd.colors = mesh.vertex_colors
    o3d.visualization.draw_geometries([final_pcd])


    # print('run Poisson surface reconstruction') # only works in complet object
    # final_pcd.estimate_normals(
    #             o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(final_pcd, depth=9)
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh])