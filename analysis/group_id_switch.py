import os, numpy as np, numba, json, yaml, argparse, multiprocessing, csv
import motmetrics as mm, mot_3d.utils as utils, mot_3d.metrics as metrics
from mot_3d.data_protos import BBox, Validity


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='baseline')
parser.add_argument('--result_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/mot_results/')
parser.add_argument('--gt_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/detection/gt/validation/')
parser.add_argument('--data_folder', type=str, default='/mnt/truenas/scratch/ziqi.pang/datasets/waymo/sot/')
parser.add_argument('--obj_type', type=str, default='vehicle')
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
    print(acc / np.sum(acc))
    return


def pred_bbox_filter(pred_bboxes, pred_ids, pred_states):
    result_bboxes, result_ids = list(), list()
    for bboxes, ids, states in zip(pred_bboxes, pred_ids, pred_states):
        indices = [i for i in range(len(states)) if Validity.valid(states[i])]
        frame_bboxes = [bboxes[i] for i in indices]
        frame_ids = [ids[i] for i in indices]
        result_bboxes.append(frame_bboxes)
        result_ids.append(frame_ids)
    return result_bboxes, result_ids


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


def sequence_eval(name, gt, pd, dist='iou', distth=0.7):
    acc = mm.MOTAccumulator()
    assert len(gt) == len(pd)

    frame_ids = np.arange(len(gt)).astype(np.int)
    wrong_types = np.array([0, 0, 0, 0])  # asso, early end, late end
    row_num = 0

    gt_id_set, pd_id_set = set(), set()
    for i, frame_id in enumerate(frame_ids):
        gt_bboxes = gt[frame_id]
        pd_bboxes = pd[frame_id]

        oids = np.array([gt_bbox[0] for gt_bbox in gt_bboxes])
        hids = np.array([pd_bbox[0] for pd_bbox in pd_bboxes])
        dists = np.empty((0, 0))
        if len(gt_bboxes) > 0 and len(pd_bboxes) > 0:
            dists = np.zeros((len(gt_bboxes), len(pd_bboxes)))
    
            for _i in range(len(gt_bboxes)):
                for _j in range(len(pd_bboxes)):
                    _, dists[_i, _j] = utils.iou3d(gt_bboxes[_i][1], pd_bboxes[_j][1])
            dists = np.where(dists >= distth, dists, np.nan)
            dists = -dists
        
        acc.update(oids, hids, dists, frameid=frame_id)
        try:
            frame_mot_events = acc.mot_events.loc[frame_id]
        except:
            continue
        id_switch_rows = frame_mot_events.loc[frame_mot_events['Type'] == 'SWITCH']
        pred_ids = id_switch_rows['HId']
        gt_ids = id_switch_rows['OId']
        row_num += len(id_switch_rows)

        if i > 0:
            try:    
                for _, (oid, hid) in enumerate(zip(gt_ids, pred_ids)):
                    if oid in gt_id_set and hid in pd_id_set:
                        wrong_types[0] += 1
                    elif oid in gt_id_set and hid not in pd_id_set:
                        wrong_types[1] += 1
                    elif oid not in gt_id_set and hid in pd_id_set:
                        wrong_types[2] += 1
                    else:
                        wrong_types[3] += 1
            except:
                pass
        
        for id in oids:
            gt_id_set.add(id)
        for id in hids:
            pd_id_set.add(id)
        
    np.savetxt(os.path.join('./tmp_data/pedestrian_ids_wo_asso', '{:}.txt'.format(name)), wrong_types)
    return wrong_types


def main(name, obj_type, distth, data_folder, gt_folder, result_folder, token, process):
    summary_folder = os.path.join(result_folder, args.src, obj_type)
    file_names = sorted(os.listdir(summary_folder))[:]
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

        # load gt
        gt_bboxes, gt_ids = load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)

        # load preds
        pred_result = np.load(os.path.join(summary_folder, file_name), allow_pickle=True)
        pred_ids = pred_result['ids']
        pred_ids = utils.id_transform(pred_ids)
        pred_bboxes = pred_result['bboxes']
        pred_states = pred_result['states']
        for i in range(len(pred_bboxes)):
            for j in range(len(pred_bboxes[i])):
                pred_bboxes[i][j] = BBox.array2bbox(pred_bboxes[i][j])
        pred_bboxes, pred_ids = pred_bbox_filter(pred_bboxes, pred_ids, pred_states)

        # dealing with the frequency
        frame_num = len(gt_bboxes)
        pred_bboxes = [pred_bboxes[i] for i in range(frame_num)]
        gt_bboxes = [gt_bboxes[i] for i in range(frame_num)]
        pred_ids = [pred_ids[i] for i in range(frame_num)]
        gt_ids = [gt_ids[i] for i in range(frame_num)]

        # evaluate
        gts = box_wrapper(bboxes=gt_bboxes, ids=gt_ids)[:]
        preds = box_wrapper(bboxes=pred_bboxes, ids=pred_ids)[:]
        accs += sequence_eval(segment_name, gts, preds, distth=distth)
        print('End {} / {}'.format(file_index + 1, len(file_names)))
    return accs


if __name__ == '__main__':
    result_folder = os.path.join(args.result_folder, args.name)
    file_names = os.listdir(os.path.join(result_folder, 'summary', args.obj_type))
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        results = pool.starmap(main, [
            (args.name, args.obj_type, args.distth, args.data_folder, args.gt_folder, result_folder, token, args.process)
            for token in range(args.process)])
        pool.close()
        pool.join()

        accs = list()
        for i in range(len(results)):
            for j in range(len(results[i])):
                accs.append(results[i][j])
    else:
        accs = main(args.name, args.obj_type, args.distth, args.data_folder, args.gt_folder, result_folder, 0, 1)
    
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
