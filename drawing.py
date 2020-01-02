# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d

import copy
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
def draw_geometries_flip(pcds):
    pcds_transform = []
    for pcd in pcds:
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.transform(flip_transform)
        pcds_transform.append(pcd_temp)
    o3d.visualization.draw_geometries(pcds_transform)

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    # pcd = o3d.io.read_point_cloud("../../TestData/fragment.ply")
    # pcd = o3d.io.read_point_cloud("dataset/realsense/scene/integrated.ply")
    pcd = o3d.io.read_point_cloud("dataset/realsense/fragments/fragment_002.ply")
    draw_geometries_flip([pcd])
    # o3d.visualization.draw_geometries_flip([pcd])
