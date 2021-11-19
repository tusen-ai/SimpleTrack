import os, open3d as o3d, numpy as np, argparse
import matplotlib.pyplot as plt
from mot_3d.visualization import Visualizer2D
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from mot_3d.data_protos import BBox
from mot_3d.utils import *
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/nuscenes/')
parser.add_argument('--vis_det', action='store_true', default=False)
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()


def o3d_pc_visualization(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    return


def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: np.ndarray = np.array([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    tm = np.eye(4)
    rotation = Quaternion(rotation)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def main(data_folder, vis_det, det_name):
    ego_pose_folder = os.path.join(data_folder, 'ego_info')
    file_names = sorted(os.listdir(ego_pose_folder))
    for file_index, file_name in enumerate(file_names):
        print('PROCESSING FILE {:} / {:}'.format(file_index + 1, len(file_names)))
        gt_info = np.load(os.path.join(data_folder, 'gt_info', file_name), allow_pickle=True)
        pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', file_name), allow_pickle=True)
        ego_poses = np.load(os.path.join(ego_pose_folder, file_name), allow_pickle=True)
        calibrations = np.load(os.path.join(data_folder, 'calib_info', file_name), allow_pickle=True)
        ids, inst_types, bboxes = gt_info['ids'], gt_info['types'], gt_info['bboxes']
        if vis_det:
            dets = np.load(os.path.join(data_folder, 'detection', det_name, 'dets', file_name), allow_pickle=True)
            bboxes, inst_types = dets['bboxes'], dets['types']
        
        frame_num = len(ids)
        for frame_index in range(frame_num):
            if frame_index % 2 != 0:
                continue
            print('FILE {:} / {:} FRAME {:}'.format(file_index + 1, len(file_names), frame_index))

            mot_bbox_list = list()
            ego = ego_poses[str(frame_index)]
            ego_trans, ego_rot = np.asarray(ego[:3]), Quaternion(np.asarray(ego[3:]))
            ego_matrix = transform_matrix(ego_trans, np.asarray(ego[3:]))

            calib = calibrations[str(frame_index)]
            calib_trans, calib_rot = np.asarray(calib[:3]), Quaternion(np.asarray(calib[3:]))

            frame_bboxes = bboxes[frame_index]
            for b in frame_bboxes:
                nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10]))
                bottom_corners = nu_box.bottom_corners()
                mot_bbox = BBox(
                    x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
                    w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
                    o=nu_box.orientation.yaw_pitch_roll[0]
                )
                mot_bbox_list.append(mot_bbox)
            
            pc = deepcopy(pcs[str(frame_index)])[:, :3]
            pc = np.dot(pc, calib_rot.rotation_matrix.T)
            pc += calib_trans
            pc = np.dot(pc, ego_rot.rotation_matrix.T)
            pc += ego_trans
            
            vis = Visualizer2D(figsize=(12, 12))
            vis.handler_pc(pc)
            for b in mot_bbox_list:
                vis.handler_box(b)
            vis.show()
            vis.close()

            # o3d_pc_visualization(np.asarray(pcs[str(frame_index)][:, :3]))
    return


if __name__ == '__main__':
    if args.test:
        data_folder = os.path.join(args.data_folder, 'test')
    elif args.mode == '2hz':
        data_folder = os.path.join(args.data_folder, 'validation')
    elif args.mode == '20hz':
        data_folder = os.path.join(args.data_folder, 'val')

    main(data_folder, args.vis_det, args.det_name)