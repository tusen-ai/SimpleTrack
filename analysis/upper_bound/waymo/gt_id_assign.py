""" Upper Bound Experiment: Assign GT IDs to bounding boxes to simulate GT Association / Life-cycle
"""
import os, argparse, numpy as np, multiprocessing,mot_3d.utils as utils
from mot_3d.data_protos import BBox


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--obj_type', type=str, default='vehicle', choices=['vehicle', 'pedestrian', 'cyclist'])
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--skip', action='store_true', default=False)
# paths
parser.add_argument('--result_name', type=str, default='gt_id')
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
    gt_ids, gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, type_field=[type_token], id_trans=False)

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


def compute_ious(preds, gts):
    if len(preds) == 0:
        return []
    if len(gts) == 0:
        return np.zeros(len(preds))
    
    ious = np.zeros((len(preds), len(gts)))
    for i, pd in enumerate(preds):
        for j, gt in enumerate(gts):
            ious[i, j] = utils.iou3d(pd, gt)[1]
    return ious


def sequence_assign(pd_bboxes, pd_ids, pd_states, gt_bboxes, gt_ids):
    frame_num = len(pd_bboxes)
    result_bboxes = []
    result_ids = []
    result_states = []
    for frame_index in range(frame_num):
        frame_pds = pd_bboxes[frame_index]
        frame_states = pd_states[frame_index]
        frame_pd_ids = pd_ids[frame_index]
        frame_gts = gt_bboxes[frame_index]
        frame_ids = gt_ids[frame_index]
        if len(frame_pds) == 0:
            result_ids.append([])
            result_states.append([])
            result_bboxes.append([])
            continue
        
        if len(frame_gts) == 0:
            result_ids.append([])
            result_states.append([])
            result_bboxes.append([])
            continue

        # compute the 3d iou matrix
        iou_matrix = compute_ious(frame_pds, frame_gts)
        max_index = np.argmax(iou_matrix, axis=1)
        max_iou = np.max(iou_matrix, axis=1)

        index = list(reversed(sorted(range(len(max_iou)), key=lambda k:max_iou[k])))
        frame_pds = [frame_pds[i] for i in index]
        frame_states = [frame_states[i] for i in index]
        frame_pd_ids = [frame_pd_ids[i] for i in index]

        frame_result_ids = list()
        for i in range(len(frame_pds)):
            iou_index = index[i]
            if max_iou[iou_index] > 0.1 and (frame_ids[max_index[iou_index]] not in frame_result_ids):
                frame_result_ids.append(frame_ids[max_index[iou_index]])
            else:
                frame_result_ids.append(frame_pd_ids[i])
            
        frame_result_pds = [frame_pds[i] for i in range(len(frame_pds)) if frame_result_ids[i] is not None]
        frame_result_states = [frame_states[i] for i in range(len(frame_states)) if frame_result_ids[i] is not None]
        frame_result_ids = [id for id in frame_result_ids if id is not None]
        result_ids.append(frame_result_ids)
        result_states.append(frame_result_states)
        result_bboxes.append(frame_result_pds)
    return result_bboxes, result_ids, result_states


def main(name, obj_type, data_folder, result_folder, gt_folder, output_folder, token=0, process=1):
    summary_folder = os.path.join(result_folder, name, 'summary', obj_type)

    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    if args.skip:
        file_names = [fname for fname in file_names if not os.path.exists(os.path.join(output_folder, fname))]

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

        result_file_path = os.path.join(summary_folder, file_name)
        result_data = np.load(result_file_path, allow_pickle=True)
        pd_ids, pd_bboxes, pd_states = \
            result_data['ids'], result_data['bboxes'], result_data['states']

        frame_num = len(pd_bboxes)
        for i in range(frame_num):
            for j in range(len(pd_bboxes[i])):
                pd_bboxes[i][j] = BBox.array2bbox(pd_bboxes[i][j])
        pd_bboxes, pd_ids, pd_states = sequence_assign(pd_bboxes, pd_ids, pd_states, gt_bboxes, gt_ids)
        for i in range(frame_num):
            for j in range(len(pd_bboxes[i])):
                pd_bboxes[i][j] = BBox.bbox2array(pd_bboxes[i][j])
        np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(segment_name)),
            ids=pd_ids, bboxes=pd_bboxes, states=pd_states)
        print('End {:} / {:}'.format(file_index + 1, len(file_names)))


if __name__ == '__main__':
    output_folder = os.path.join(args.result_folder, args.result_name, 'summary', args.obj_type)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(
                args.name, args.obj_type, args.data_folder, args.result_folder, 
                args.gt_folder, output_folder, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.name, args.obj_type, args.data_folder, args.result_folder, args.gt_folder, output_folder)