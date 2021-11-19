import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import WaymoLoader


parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--det_name', type=str, default='public')
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
parser.add_argument('--obj_type', type=str, default='vehicle', choices=['vehicle', 'pedestrian', 'cyclist'])
parser.add_argument('--verbose', action='store_true', default=False)
# paths
parser.add_argument('--config_path', type=str, default='../configs/config.yaml')
parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/mot_results/')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--det_data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/')
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
    return gt_bboxes, gt_ids


def main(name, obj_type, config_path, data_folder, det_data_folder, result_folder, gt_folder, start_frame=0, token=0, process=1):
    summary_folder = os.path.join(result_folder, 'summary', obj_type)
    # simply knowing about all the segments
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    if args.skip:
        file_names = [fname for fname in file_names if not os.path.exists(os.path.join(summary_folder, fname))]
    
    # load model configs
    configs = yaml.load(open(config_path, 'r'))
    
    cnt = 0
    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4
    for file_index, file_name in enumerate(file_names[:]):
        if file_index % process != token:
            continue
        print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]
        gt_bboxes, gt_ids = load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)

        for gts in gt_bboxes:
            cnt += len(gts)
    
    print(cnt)


if __name__ == '__main__':
    result_folder = os.path.join(args.result_folder, args.name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    summary_folder = os.path.join(result_folder, 'summary')
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    summary_folder = os.path.join(summary_folder, args.obj_type)
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    det_data_folder = os.path.join(args.det_data_folder, args.det_name)

    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.name, args.obj_type, args.config_path, args.data_folder, det_data_folder, 
                result_folder, args.gt_folder, 0, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.name, args.obj_type, args.config_path, args.data_folder, det_data_folder, result_folder, 
            args.gt_folder, args.start_frame, 0, 1)