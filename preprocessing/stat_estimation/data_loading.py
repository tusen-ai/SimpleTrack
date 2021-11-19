import os, numpy as np, mot_3d, mot_3d.utils as utils
from mot_3d.data_protos import BBox
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box


__all__ = ['load_gt_bboxes', 'load_dets']


def nu_array2mot_bbox(b):
    nu_box = Box(b[:3], b[3:6], Quaternion(b[6:10]))
    mot_bbox = BBox(
        x=nu_box.center[0], y=nu_box.center[1], z=nu_box.center[2],
        w=nu_box.wlh[0], l=nu_box.wlh[1], h=nu_box.wlh[2],
        o=nu_box.orientation.yaw_pitch_roll[0]
    )
    if len(b) == 11:
        mot_bbox.s = b[-1]
    return mot_bbox


def load_gt_bboxes(gt_folder, data_folder, segment_name, type_token, dataset='waymo'):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']
    
    frame_num = len(bboxes)
    if dataset == 'nuscenes':
        for i in range(frame_num):
            for j in range(len(bboxes[i])):
                bboxes[i][j] = BBox.bbox2array(nu_array2mot_bbox(bboxes[i][j]))
        for i in range(frame_num):
            for j in range(len(inst_types[i])):
                try:
                    if type_token in inst_types[i][j]:
                        inst_types[i][j] = type_token
                except:
                    print(len(inst_types[i]), j)
                    exit(0)

    gt_ids, gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, type_field=[type_token], id_trans=True)
    if dataset == 'waymo':
        ego_keys = sorted(utils.str2int(ego_info.keys()))
        egos = [ego_info[str(key)] for key in ego_keys]
        gt_bboxes = bboxes2world(gt_bboxes, egos)
    return gt_bboxes, gt_ids


def load_dets(det_data_folder, data_folder, segment_name, type_token, dataset='waymo'):
    dets_info = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    dets_bboxes = dets_info['bboxes']
    inst_types = dets_info['types']

    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    
    det_bboxes = [dets_bboxes[key] for key in ego_keys]
    inst_types = [inst_types[key] for key in ego_keys]
    dets = [[] for i in range(len(dets_bboxes))]
    for i in range(len(det_bboxes)):
        for j in range(len(det_bboxes[i])):
            if inst_types[i][j] == type_token:
                if dataset == 'nuscenes':
                    bbox = nu_array2mot_bbox(dets_bboxes[i][j])
                else:
                    bbox = BBox.array2bbox(dets_bboxes[i][j])
                dets[i].append(bbox)
    if dataset == 'waymo':
        dets = bboxes2world(dets, egos)
    return dets


def bboxes2world(bboxes, egos):
    frame_num = len(egos)
    for i in range(frame_num):
        ego = egos[i]
        bbox_num = len(bboxes[i])
        for j in range(bbox_num):
            bboxes[i][j] = BBox.bbox2world(ego, bboxes[i][j])
    return bboxes
