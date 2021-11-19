import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
import stat_estimation as se
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='')
parser.add_argument('--det_name', type=str, default='cp_nms')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--det_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--frequency', type=int, default=1)
parser.add_argument('--result_folder', type=str, default='./waymo_data/waymo_stats/')
parser.add_argument('--part', type=str, default='measurement', choices=['process', 'measurement'])
args = parser.parse_args()


OBJ_TYPES = ['vehicle', 'pedestrian', 'cyclist']


def get_process_stats(data_folder, gt_folder, obj_type, frequency):
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    result_bbox_data = list()

    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4

    print('Processing for type ', obj_type)
    # aggregate the bbox arrays from all the sequences
    pbar = tqdm(total=len(file_names))
    for file_index, file_name in enumerate(file_names[:2]):
        segment_name = file_name.split('.')[0]

        gt_bboxes, gt_ids = se.load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)
        gt_bboxes = [gt_bboxes[i] for i in range(len(gt_bboxes)) if i % frequency == 0]
        gt_ids = [gt_ids[i] for i in range(len(gt_ids)) if i % frequency == 0]
        sequence_bbox_data = se.process_stats(gt_bboxes, gt_ids)
        result_bbox_data.append(sequence_bbox_data)
        pbar.update(1)
    pbar.close()
    # compute the variance and convert to matrix Q diagnals
    result_bbox_data = np.vstack(result_bbox_data)
    var = np.var(result_bbox_data, axis=0)
    q_array = var[11:].tolist() + [0, 0, 0] + var[11:].tolist()
    return q_array


def get_measurement_stats(data_folder, gt_folder, det_folder, obj_type, frequency):
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    result_diff, result_diff_vel = list(), list()

    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4
    
    print('Processing for type ', obj_type)
    pbar = tqdm(total=len(file_names))
    for file_index, file_name in enumerate(file_names[:]):
        segment_name = file_name.split('.')[0]

        gt_bboxes, gt_ids = se.load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)
        det_bboxes = se.load_dets(det_folder, data_folder, segment_name, type_token)
        gt_bboxes = [gt_bboxes[i] for i in range(len(gt_bboxes)) if i % frequency == 0]
        gt_ids = [gt_ids[i] for i in range(len(gt_ids)) if i % frequency == 0]
        det_bboxes = [det_bboxes[i] for i in range(len(det_bboxes)) if i % frequency == 0]
    
        diff, diff_vel = se.measurement_stats(det_bboxes, gt_bboxes, gt_ids)
        result_diff.append(diff)
        result_diff_vel.append(diff_vel)
        pbar.update(1)
    pbar.close()

    # compute the variance of value and velocity
    result_diff = np.vstack(result_diff)
    result_diff_vel = np.vstack(result_diff_vel)
    diff_var = np.var(result_diff, axis=0)
    diff_vel_var = np.var(result_diff_vel, axis=0)

    p_array = diff_var.tolist() + diff_vel_var[-4:].tolist()
    r_array = diff_var.tolist()
    return p_array, r_array


if __name__ == '__main__':
    if args.part == 'process':
        result = dict()
        for obj_type in OBJ_TYPES:
            q_array = get_process_stats(args.data_folder, args.gt_folder, obj_type, args.frequency)
            result[obj_type] = q_array
        
        f = open(os.path.join(args.result_folder, 'Q_{:}.json'.format(args.name)), 'w')
        json.dump(result, f)
        f.close()
    elif args.part == 'measurement':
        result_p, result_r = dict(), dict()
        det_folder = os.path.join(args.det_folder, args.det_name)
        for obj_type in OBJ_TYPES:
            p_array, r_array = get_measurement_stats(args.data_folder, args.gt_folder, det_folder,
                obj_type, args.frequency)
            result_p[obj_type] = p_array
            result_r[obj_type] = r_array
        
        f = open(os.path.join(args.result_folder, 'P_{:}.json'.format(args.name)), 'w')
        json.dump(result_p, f)
        f.close()

        f = open(os.path.join(args.result_folder, 'R_{:}.json'.format(args.name)), 'w')
        json.dump(result_r, f)
        f.close()