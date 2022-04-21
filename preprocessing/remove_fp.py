import os, argparse, numpy as np, multiprocessing
import mot_3d, mot_3d.utils as utils, mot_3d.visualization as visualization
from nuscenes.utils.data_classes import Box
from mot_3d.data_protos import BBox
from pyquaternion import Quaternion
from  preprocessing.detection_nms import load_dets, load_pcs, bbox_array2nuscenes_format, nu_array2mot_bbox
from preprocessing.gt_association.associate import associate

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='remove_fp')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--dist_thres', type=float, default=2.0)
parser.add_argument('--dist_type', type=str, default='l2')
parser.add_argument('--data_folder', type=str, default='/media/colton/ColtonSSD/simpletrack-nuscenes/validation_2hz')
parser.add_argument('--gt_folder', type=str,
                    default='/media/colton/ColtonSSD/simpletrack-nuscenes/validation_2hz/gt_info')
parser.add_argument('--det_folder', type=str, default='/media/colton/ColtonSSD/simpletrack-nuscenes/validation_2hz/detection/')
parser.add_argument('--visualize', action='store_true', default=False)
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--dataset', type=str, default='waymo', choices=['waymo', 'nuscenes'])
args = parser.parse_args()

def load_gt_bboxes(gt_folder, data_folder, segment_name):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)),
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)),
        allow_pickle=True)

    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']

    frame_num = len(bboxes)
    for i in range(frame_num):
        for j in range(len(bboxes[i])):
            if args.dataset == 'nuscenes':
                bboxes[i][j] = nu_array2mot_bbox(bboxes[i][j])
            else:
                bboxes[i][j] = BBox.array2bbox(bboxes[i][j])

    return bboxes, ids, inst_types


def main(det_name, thres, data_folder, gt_folder, det_folder, output_folder, token, process):
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))

    for file_index, file_name in enumerate(file_names[:]):
        if file_index % process != token:
            continue
        print('SEQ {:} / {:}'.format(file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]
        dets, inst_types = load_dets(os.path.join(det_folder, det_name), data_folder, segment_name)
        gt_bboxes, _, gt_inst_types = load_gt_bboxes(gt_folder, data_folder, segment_name)
        if args.visualize:
            pcs = load_pcs(data_folder, segment_name)

        frame_num = len(inst_types)
        result_bboxes = list()
        result_types = list()
        for frame_index in range(0, frame_num):
            frame_dets = dets[frame_index]
            frame_types = inst_types[frame_index]
            frame_gt = gt_bboxes[frame_index]
            frame_gt_types = gt_inst_types[frame_index]

            # perform gt association
            tp, tp_matches, fp, fn, tp_types, fp_types, fn_types = associate(frame_gt, frame_gt_types, frame_dets, frame_types, threshold=thres)

            # visualize
            if args.visualize:
                vis = visualization.Visualizer2D(name='{:}'.format(frame_index), figsize=(20, 20))
                # vis.handler_pc(pcs[frame_index])
                for i, det in enumerate(tp_matches):
                    vis.handler_box(det, '', color='black', linestyle='solid')
                for i, det in enumerate(tp):
                    vis.handler_box(det, '%.2f' % det.s, color='light_blue', linestyle='dashed')
                # for i, det in enumerate(fp):
                #     vis.handler_box(det, '', color='red')
                vis.show()
                vis.close()

            if args.dataset == 'waymo':
                frame_bboxes = [BBox.bbox2array(det) for det in tp]
            elif args.dataset == 'nuscenes':
                frame_bboxes = [bbox_array2nuscenes_format(BBox.bbox2array(det)) for det in tp]
            frame_types = tp_types
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
            result = pool.apply_async(main, args=(
            args.det_name, args.thres, args.data_folder, args.gt_folder, args.det_folder,
            output_det_folder, token, args.process))
        pool.close()
        pool.join()
    else:
        main(args.det_name, args.dist_thres, args.data_folder, args.gt_folder, args.det_folder, output_det_folder, 0, 1)