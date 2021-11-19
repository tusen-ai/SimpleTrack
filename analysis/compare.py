""" compare the difference between two sequences
"""
import os, argparse, multiprocessing, json, numpy as np
import mot_3d.utils as utils, mot_3d.visualization as vis
from mot_3d.data_protos import BBox, Validity
from contrast.error import Displayer


parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, default='cp_rkf,cp_nms_iou')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--det_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/cp_nms/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/dets/')
parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/mot_results/')
parser.add_argument('--obj_type', type=str, default='vehicle', choices=['vehicle', 'pedestrian', 'cyclist'])
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--skip', action='store_true', default=False)
args = parser.parse_args()


def load_gt_bboxes(gt_folder, data_folder, segment_name, type_token):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']
    
    gt_ids, gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, type_field=[type_token], id_trans=True)

    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    gt_bboxes = gt_bbox2world(gt_bboxes, egos)
    return gt_bboxes, gt_ids


def gt_bbox2world(bboxes, egos):
    frame_num = len(egos)
    for i in range(frame_num):
        ego = egos[i]
        bbox_num = len(bboxes[i])
        for j in range(bbox_num):
            bboxes[i][j] = BBox.bbox2world(ego, bboxes[i][j])
    return bboxes


def main(names, obj_type, data_folder, det_folder, result_folder, gt_folder):
    ego_folder = os.path.join(data_folder, 'ego_info')
    file_names = sorted(os.listdir(ego_folder))
    name0, name1 = names.split(',')
    summary_folder0 = os.path.join(result_folder, name0, 'summary', obj_type)
    summary_folder1 = os.path.join(result_folder, name1, 'summary', obj_type)

    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4

    for file_index, file_name in enumerate(file_names[:]):
        print('Eval {} / {}'.format(file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]

        displayer = Displayer(namea=name0, nameb=name1, obj_type=obj_type, seq_id=segment_name, data_folder=data_folder, result_folder=result_folder,
            det_data_folder=det_folder, gt_folder=gt_folder)

        gt_bboxes, _ = load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)
        frame_num = len(gt_bboxes)
        
        frame_index = 0
        while frame_index < frame_num:
            print('SEQ {:} FRAME {:} / {:}'.format(file_index, frame_index, frame_num))
            displayer.single_frame_display(frame_index)
            command = input('Frame Control:')
            if 'a' in command:
                frame_index -= 1
                continue
            else:
                frame_index += 1
    return


if __name__ == '__main__':
    main(args.names, args.obj_type, args.data_folder, args.det_folder, args.result_folder, args.gt_folder)