import os, argparse, numpy as np, multiprocessing
import mot_3d, mot_3d.utils as utils, mot_3d.visualization as visualization
from nuscenes.utils.data_classes import Box
from mot_3d.data_protos import BBox
from pyquaternion import Quaternion
import py_nms


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nms')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--thres', type=float, default=0.1)
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--det_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/')
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='waymo', choices=['waymo', 'nuscenes'])
args = parser.parse_args()


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


def bbox_array2nuscenes_format(bbox_array):
    translation = bbox_array[:3].tolist()
    size = bbox_array[4:7].tolist()
    size = [size[1], size[0], size[2]]
    score = bbox_array[-1]

    yaw = bbox_array[3]
    rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                           [np.sin(yaw),  np.cos(yaw), 0, 0],
                           [0,            0,           1, 0],
                           [0,            1,           0, 1]])
    q = Quaternion(matrix=rot_matrix)
    rotation = q.q.tolist()

    result = translation + size + rotation + [score]
    return result


def load_gt_bboxes(gt_folder, data_folder, segment_name):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']

    if args.dataset == 'nuscenes':
        frame_num = len(bboxes)
        for i in range(frame_num):
            for j in range(len(bboxes[i])):
                bboxes[i][j] = BBox.bbox2array(nu_array2mot_bbox(bboxes[i][j]))

    gt_ids, gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, id_trans=True)
    gt_ids, gt_bboxes = gt_ids[0], gt_bboxes[0]
    return gt_bboxes, gt_ids


def load_pcs(data_folder, segment_name):
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    
    if args.dataset == 'waymo':
        pcs = np.load(os.path.join(data_folder, 'pc', 'clean_pc', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
    elif args.dataset == 'nuscenes':
        pcs = np.load(os.path.join(data_folder, 'pc', 'raw_pc', '{:}.npz'.format(segment_name)), 
            allow_pickle=True)
    result = [pcs[str(key)] for key in ego_keys]
    return result


def load_dets(det_data_folder, data_folder, segment_name):
    dets_info = np.load(os.path.join(det_data_folder, 'dets', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    dets_bboxes = dets_info['bboxes']
    inst_types = dets_info['types']

    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    
    if args.dataset == 'waymo':      
        det_bboxes = [dets_bboxes[key] for key in ego_keys]
        inst_types = [inst_types[key] for key in ego_keys]
        dets = [[] for i in range(len(dets_bboxes))]
        for i in range(len(det_bboxes)):
            for j in range(len(det_bboxes[i])):
                dets[i].append(BBox.array2bbox(dets_bboxes[i][j]))
    elif args.dataset == 'nuscenes':
        det_bboxes = [dets_bboxes[key] for key in ego_keys]
        inst_types = [inst_types[key] for key in ego_keys]
        dets = [[] for i in range(len(dets_bboxes))]
        for i in range(len(det_bboxes)):
            for j in range(len(det_bboxes[i])):
                dets[i].append(nu_array2mot_bbox(dets_bboxes[i][j]))
    return dets, inst_types


def main(det_name, thres, data_folder, gt_folder, det_folder, output_folder, token, process):
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    if args.skip:
        file_names = [fname for fname in file_names if not os.path.exists(os.path.join(output_folder, 'dets', fname))]
    
    for file_index, file_name in enumerate(file_names[:]):
        if file_index % process != token:
            continue
        print('SEQ {:} / {:}'.format(file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]
        dets, inst_types = load_dets(os.path.join(det_folder, det_name), data_folder, segment_name)
        if args.visualize:
            gt_bboxes, _ = load_gt_bboxes(gt_folder, data_folder, segment_name)
            pcs = load_pcs(data_folder, segment_name)

        frame_num = len(inst_types)
        result_bboxes = list()
        result_types = list()
        for frame_index in range(0, frame_num):
            frame_dets = dets[frame_index]
            frame_types = inst_types[frame_index]
            
            # nms
            nms_dets, nms_types = py_nms.nms(frame_dets, frame_types, threshold_low=thres, threshold_high=1.0)
            if args.visualize:            
                vis = visualization.Visualizer2D(name='{:}'.format(frame_index), figsize=(20, 20))
                # vis.handler_pc(pcs[frame_index])
                for i, det in enumerate(frame_dets):
                    vis.handler_box(det, '%.2f' % det.s, color='light_blue', linestyle='dashed')
                for i, det in enumerate(nms_dets):
                    vis.handler_box(det, '%s %.2f' % (str(nms_types[i]), det.s), color='red')
                vis.show()
                vis.close()
            
            if args.dataset == 'waymo':
                frame_bboxes = [BBox.bbox2array(det) for det in nms_dets]
            elif args.dataset == 'nuscenes':
                frame_bboxes = [bbox_array2nuscenes_format(BBox.bbox2array(det)) for det in nms_dets]
            frame_types = nms_types
            result_bboxes.append(frame_bboxes)
            result_types.append(frame_types)

            if (frame_index + 1) % 1 == 0:
                print('SEQ {:} FRAME {:} / {:}'.format(file_index + 1, frame_index + 1, frame_num))
        
        np.savez_compressed(os.path.join(output_folder, 'dets', '{:}.npz'.format(segment_name)),
            types=result_types, bboxes=result_bboxes)
    return


if __name__ == '__main__':
    output_det_folder = os.path.join(args.det_folder, args.name)
    if not os.path.exists(output_det_folder):
        os.makedirs(output_det_folder)
    if not os.path.exists(os.path.join(output_det_folder, 'dets')):
        os.makedirs(os.path.join(output_det_folder, 'dets'))
    
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(args.det_name, args.thres, args.data_folder, args.gt_folder, args.det_folder, 
                output_det_folder, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.det_name, args.thres, args.data_folder, args.gt_folder, args.det_folder, output_det_folder, 0, 1)