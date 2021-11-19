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
parser.add_argument('--config_path', type=str, default='configs/config.yaml')
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


def frame_visualization(bboxes, ids, states, gt_bboxes=None, gt_ids=None, pc=None, dets=None, name=''):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
    if pc is not None:
        visualizer.handler_pc(pc)
    for _, bbox in enumerate(gt_bboxes):
        visualizer.handler_box(bbox, message='', color='black')
    dets = [d for d in dets if d.s >= 0.1]
    for det in dets:
        visualizer.handler_box(det, message='%.2f' % det.s, color='purple', linestyle='dashed')
    for _, (bbox, id, state) in enumerate(zip(bboxes, ids, states)):
        if Validity.valid(state):
            visualizer.handler_box(bbox, message=str(id), color='red')
        else:
            visualizer.handler_box(bbox, message=str(id), color='light_blue')
    # visualizer.show()
    visualizer.save(os.path.join('imgs', '{:}.png'.format(name)))
    visualizer.close()
    # result = {}
    # result['pc'] = pc
    # result['gt'] = [BBox.bbox2array(b) for b in gt_bboxes]
    # result['bboxes'] = [BBox.bbox2array(b) for b in bboxes]
    # result['ids'] = ids
    # result['states'] = states
    # np.savez_compressed(os.path.join('imgs', 'giou', '{:}.npz'.format(name)), **result)


def sequence_mot(configs, data_loader: WaymoLoader, sequence_id, gt_bboxes=None, gt_ids=None, visualize=False):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    IDs, bboxes, states, types = list(), list(), list(), list()
    for frame_index in range(data_loader.cur_frame, frame_num):
        print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(data_loader.type_token, sequence_id + 1, frame_index + 1, frame_num))
        # input data
        frame_data = next(data_loader)
        frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'], 
            det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'])
        # frame_data.dets = utils.score_rectification(frame_data.dets, gt_bboxes[frame_index])

        # mot
        results = tracker.frame_mot(frame_data)
        result_pred_bboxes = [trk[0] for trk in results]
        result_pred_ids = [trk[1] for trk in results]
        result_pred_states = [trk[2] for trk in results]
        result_types = [trk[3] for trk in results]

        # visualization
        if visualize:
            frame_visualization(result_pred_bboxes, result_pred_ids, result_pred_states,
                gt_bboxes[frame_index], gt_ids[frame_index], frame_data.pc, dets=frame_data.dets, name='{:}_{:}_{:}'.format(args.name, sequence_id, frame_index))
        
        # wrap for output
        IDs.append(result_pred_ids)
        result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)
    return IDs, bboxes, states, types


def main(name, obj_type, config_path, data_folder, det_data_folder, result_folder, gt_folder, start_frame=0, token=0, process=1):
    summary_folder = os.path.join(result_folder, 'summary', obj_type)
    # simply knowing about all the segments
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    if args.skip:
        file_names = [fname for fname in file_names if not os.path.exists(os.path.join(summary_folder, fname))]
    
    # load model configs
    configs = yaml.load(open(config_path, 'r'))
    
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
        data_loader = WaymoLoader(configs, [type_token], segment_name, data_folder, det_data_folder, start_frame)
        gt_bboxes, gt_ids = load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)

        # real mot happens here
        ids, bboxes, states, types = sequence_mot(configs, data_loader, file_index, gt_bboxes, gt_ids, args.visualize)
        np.savez_compressed(os.path.join(summary_folder, '{}.npz'.format(segment_name)),
            ids=ids, bboxes=bboxes, states=states)


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
    
