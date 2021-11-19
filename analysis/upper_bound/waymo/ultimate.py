import os, argparse, numpy as np, multiprocessing,mot_3d.utils as utils
from mot_3d.data_protos import BBox


parser = argparse.ArgumentParser()
parser.add_argument('--result_name', type=str, default='ultimate')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--obj_type', type=str, default='vehicle', choices=['vehicle', 'pedestrian', 'cyclist'])
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--thres', type=float, default=0.7)
# paths
parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/mot_results/')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--det_data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/')
args = parser.parse_args()


def load_gt_bboxes(gt_folder, data_folder, segment_name, type_token):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']
    gt_ids, gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, type_field=[type_token], id_trans=False)

    return gt_bboxes, gt_ids


def load_dets(det_data_folder, data_folder, segment_name, type_token, dataset='waymo'):
    dets_info = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    dets_bboxes = dets_info['bboxes']
    inst_types = dets_info['types']

    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_keys = sorted(utils.str2int(ego_info.keys()))
    
    det_bboxes = [dets_bboxes[key] for key in ego_keys]
    inst_types = [inst_types[key] for key in ego_keys]
    dets = [[] for i in range(len(dets_bboxes))]
    for i in range(len(det_bboxes)):
        for j in range(len(det_bboxes[i])):
            if inst_types[i][j] == type_token:
                bbox = BBox.array2bbox(dets_bboxes[i][j])
                dets[i].append(bbox)
    return dets


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


def sequence_match(dets, gts, gt_ids, sequence_token=''):
    frame_num = len(dets)
    result_pds, result_states, result_ids = [], [], []
    for frame_index in range(frame_num):
        print('SEQ {:} frame {:} / {:}'.format(sequence_token, frame_index + 1, frame_num))
        frame_dets = dets[frame_index]
        frame_gts = gts[frame_index]
        frame_gt_ids = gt_ids[frame_index]

        index = list(reversed(sorted(range(len(frame_dets)), key=lambda k:frame_dets[k].s)))
        frame_dets = [frame_dets[i] for i in index]

        if len(frame_dets) == 0 or len(frame_gts) == 0:
            result_pds.append([])
            result_states.append([])
            result_ids.append([])
            continue
        
        result_frame_pds, result_frame_states, result_frame_ids = [], [], []
        
        # compute the 3d iou matrix
        iou_matrix = compute_ious(frame_dets, frame_gts)
        max_index = np.argmax(iou_matrix, axis=1)
        max_iou = np.max(iou_matrix, axis=1)

        for i in range(len(frame_dets)):
            if max_iou[i] >= args.thres and frame_gt_ids[max_index[i]] not in result_frame_ids:
                result_frame_ids.append(frame_gt_ids[max_index[i]])
                result_frame_pds.append(frame_gts[max_index[i]])
                result_frame_states.append('alive_1_1')
        result_ids.append(result_frame_ids)
        result_pds.append(result_frame_pds)
        result_states.append(result_frame_states)
    return result_pds, result_ids, result_states


def main(det_name, obj_type, data_folder, result_folder, gt_folder, output_folder, token=0, process=1):
    det_folder = os.path.join(data_folder, 'detection', det_name)
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

        dets = load_dets(det_folder, data_folder, segment_name, type_token)
        frame_num = len(dets)

        pd_bboxes, pd_ids, pd_states = sequence_match(dets, gt_bboxes, gt_ids, file_index + 1)
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
                args.det_name, args.obj_type, args.data_folder, args.result_folder, 
                args.gt_folder, output_folder, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.det_name, args.obj_type, args.data_folder, args.result_folder, args.gt_folder, output_folder)