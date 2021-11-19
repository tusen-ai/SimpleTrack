import numpy as np, os, argparse
import mot_3d.utils as utils
from mot_3d.data_protos import BBox
from copy import deepcopy
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--src_name', type=str, default='cp')
parser.add_argument('--tgt_name', type=str, default='cp_score')
parser.add_argument('--det_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--process', type=int, default=1)
args = parser.parse_args()


if not os.path.exists(os.path.join(args.det_folder, args.tgt_name, 'dets')):
    os.makedirs(os.path.join(args.det_folder, args.tgt_name, 'dets'))


def load_gt_bboxes(gt_folder, segment_name):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']
    return bboxes, ids


def main(src_folder, tgt_folder, data_folder, gt_folder, token, process):
    file_names = sorted(os.listdir(src_folder))
    for file_index, file_name in enumerate(file_names):
        if file_index % process != token:
            continue
        segment_name = file_name.split('.')[0]
        src_dets = np.load(os.path.join(src_folder, '{:}.npz'.format(segment_name)),
            allow_pickle=True)
        gt_bboxes, _ = load_gt_bboxes(gt_folder, segment_name)
        result = deepcopy(src_dets['bboxes'])
        dets = deepcopy(src_dets['bboxes'])

        frame_num = len(gt_bboxes)
        for frame_index in range(frame_num):
            frame_dets = dets[frame_index]
            frame_gts = gt_bboxes[frame_index]

            if len(frame_dets) == 0 or len(frame_gts) == 0:
                continue
            iou_matrix = np.zeros((len(frame_dets), len(frame_gts)))
            for i, d in enumerate(frame_dets):
                for j, g in enumerate(frame_gts):
                    iou_matrix[i, j] = utils.iou3d(
                        BBox.array2bbox(d), BBox.array2bbox(g))[1]
            max_iou = np.max(iou_matrix, axis=1)

            for i, _ in enumerate(frame_dets):
                result[frame_index][i][-1] = max_iou[i]
            print('SEQ {:} FRAME {:} / {:}'.format(file_index + 1, frame_index + 1, frame_num))

        np.savez_compressed(os.path.join(tgt_folder, file_name), bboxes=result, types=src_dets['types'])


if __name__ == '__main__':
    src_folder = os.path.join(args.det_folder, args.src_name, 'dets')
    tgt_folder = os.path.join(args.det_folder, args.tgt_name, 'dets')
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(main, args=(src_folder, tgt_folder, args.data_folder, args.gt_folder, 
                token, args.process))
        pool.close()
        pool.join()
    else:
        main(src_folder, tgt_folder, args.data_folder, args.gt_folder, 
                0, 1)
