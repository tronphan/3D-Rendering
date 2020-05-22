# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d

import copy

def draw_geometries_flip(pcds):
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    pcds_transform = []
    for pcd in pcds:
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.transform(flip_transform)
        pcds_transform.append(pcd_temp)
    o3d.visualization.draw_geometries(pcds_transform)

def draw_registration_result_original_color(source, target, transformation=np.identity(4)):
    # if transformation == None: transformation = np.identity(4)
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def draw_registration_result(source, target, transformation=np.identity(4)):
    # if transformation == None: transformation = 
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

if __name__ == "__main__":
    print("Load a ply point cloud and render it")
    pcd = o3d.io.read_point_cloud("dataset/realsense/scene/integrated.ply")
    draw_geometries_flip([pcd])