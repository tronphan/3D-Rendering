import numpy as np
import quaternion
import pandas as pd
import json
import open3d as o3d

import os, sys
sys.path.append("config")
from initialize_config import initialize_config
from T265_registration import create_RGBD_point_cloud

def get_pose_matrix(pose):
    # Compute the transformation matrix
    q = np.quaternion(pose["Rot w"], pose["Rot x"], pose["Rot y"], pose["Rot z"])
    rotation = quaternion.as_rotation_matrix(q)
    translation = np.array([pose["Pos x"], pose["Pos y"], pose["Pos z"]])
    pose_matrix = np.identity(4)
    pose_matrix[:3,:3] = rotation
    pose_matrix[:3,3] = translation
    return pose_matrix

if __name__ == "__main__":
    print("##############################################")
    print("1. Load 3 point clouds")
    print("##############################################")
    # Load RGBD file from intel D435
    pcds_file_number = [300, 700, 750]
    with open("config/realsense.json") as json_file:
        config = json.load(json_file)
        initialize_config(config)
    pcds = [create_RGBD_point_cloud(file_number,config) for file_number in pcds_file_number]
    flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]  
    pcds = [pcd.transform(flip) for pcd in pcds]

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
