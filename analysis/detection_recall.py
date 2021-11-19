import os, numpy as np, numba, json, yaml, argparse, multiprocessing, csv
import motmetrics as mm, mot_3d.utils as utils, mot_3d.metrics as metrics
from mot_3d.data_protos import BBox, Validity


parser = argparse.ArgumentParser()
parser.add_argument('--det_name', type=str, default='baseline')
parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/mot_results/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--obj_type', type=str, default='vehicle')
parser.add_argument('--det_data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/')
parser.add_argument('--src', type=str, default='summary')
parser.add_argument('--process', type=int, default=1)
parser.add_argument('--distth', type=float, default=0.7)
args = parser.parse_args()


def folder_results(folder):
    file_names = os.listdir(folder)
    acc = np.zeros(4)
    for fname in file_names:
        data = np.loadtxt(os.path.join(folder, fname))
        acc += data
    
    print(acc)
    return


def inst_num(ids):
    result = 0
    for i in range(len(ids)):
        result += len(ids[i])
    return result


def box_wrapper(bboxes, ids):
    frame_num = len(ids)
    result = list()
    for _i in range(frame_num):
        frame_result = list()
        num = len(ids[_i])
        for _j in range(num):
            frame_result.append((ids[_i][_j], bboxes[_i][_j]))
        result.append(frame_result)
    return result


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


def sequence_eval(name, gt, pd, dist='iou', distth=0.7):
    acc = mm.MOTAccumulator()
    assert len(gt) == len(pd)

    frame_ids = np.arange(len(gt)).astype(np.int)
    nms_metrics = np.array([0, 0, 0, 0])  # gt_all, det_all, gt_match
    row_num = 0

    final_path = os.path.join('./tmp_data/cp_nms_vehicle_recall', '{:}.txt'.format(name))
    if os.path.exists(final_path):
        return nms_metrics

    for i, frame_id in enumerate(frame_ids):
        gt_bboxes = gt[frame_id]
        pd_bboxes = pd[frame_id]

        dists = np.empty((0, 0))
        if len(gt_bboxes) > 0 and len(pd_bboxes) > 0:
            dists = np.zeros((len(gt_bboxes), len(pd_bboxes)))
    
            for _i in range(len(gt_bboxes)):
                for _j in range(len(pd_bboxes)):
                    _, dists[_i, _j] = utils.iou3d(gt_bboxes[_i], pd_bboxes[_j])
            max_iou = np.max(dists, axis=1)
            gt_match = np.sum(max_iou > args.distth)
            nms_metrics[2] += gt_match

        nms_metrics[0] += len(gt_bboxes)
        nms_metrics[1] += len(pd_bboxes)

    np.savetxt(final_path, nms_metrics)
    return nms_metrics


def main(det_folder, obj_type, distth, data_folder, gt_folder, result_folder, token, process):
    file_names = sorted(os.listdir(det_folder))[:]
    accs = np.array([0, 0, 0, 0])
    
    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4
    
    for file_index, file_name in enumerate(file_names[:]):
        if file_index % process != token:
            continue
        print('Eval {} / {}'.format(file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]
        final_path = os.path.join('./tmp_data/cp_nms_vehicle_recall', '{:}.txt'.format(segment_name))
        if os.path.exists(final_path):
            continue

        # load gt
        gt_bboxes, gt_ids = load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)

        # load preds
        pred_result = np.load(os.path.join(det_folder, file_name), allow_pickle=True)
        pred_bboxes = pred_result['bboxes']
        result_pred_bboxes = [[] for i in range(len(pred_bboxes))]
        for i in range(len(pred_bboxes)):
            for j in range(len(pred_bboxes[i])):
                result_pred_bboxes[i].append(BBox.array2bbox(pred_bboxes[i][j]))
        pred_bboxes = result_pred_bboxes

        # dealing with the frequency
        frame_num = len(gt_bboxes)
        pred_bboxes = [pred_bboxes[i] for i in range(frame_num)]
        gt_bboxes = [gt_bboxes[i] for i in range(frame_num)]

        # evaluate
        accs += sequence_eval(segment_name, gt_bboxes, pred_bboxes, distth=distth)
        print('End {} / {}'.format(file_index + 1, len(file_names)))
    return accs


if __name__ == '__main__':
    result_folder = ''
    det_folder = os.path.join(args.det_data_folder, args.det_name, 'dets')
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        results = pool.starmap(main, [
            (det_folder, args.obj_type, args.distth, args.data_folder, args.gt_folder, result_folder, token, args.process)
            for token in range(args.process)])
        pool.close()
        pool.join()

        accs = list()
        for i in range(len(results)):
            for j in range(len(results[i])):
                accs.append(results[i][j])
    else:
        accs = main(det_folder, args.obj_type, args.distth, args.data_folder, args.gt_folder, result_folder, 0, 1)
    
    # mh = mm.metrics.create()
    # metrics = list(mm.metrics.motchallenge_metrics)
    # summary = mh.compute_many(accs, file_names[:], metrics,generate_overall=True)
    # summary_text = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    # print(summary_text)

    # f = open(os.path.join(result_folder, '{:}_{:}_{:}.txt'.format(args.name, args.mode, args.src)), 'w')
    # f.write(summary_text)
    # f.close()

    result = np.zeros(4)
    for acc in accs:
        result += acc
    print(result)
