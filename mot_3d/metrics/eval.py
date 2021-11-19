from collections import OrderedDict
import glob, os, numpy as np, motmetrics as mm
from ..data_protos import BBox
from ..utils import iou3d, iou2d


def eval_sequence_core(gt, pd, dist='iou', distth=0.7):
    acc = mm.MOTAccumulator()
    assert len(gt) == len(pd)

    frame_ids = np.arange(len(gt)).astype(np.int)
    names = list()
    for frame_id in frame_ids:
        gt_bboxes = gt[frame_id]
        pd_bboxes = pd[frame_id]

        oids = np.array([gt_bbox[0] for gt_bbox in gt_bboxes])
        hids = np.array([pd_bbox[0] for pd_bbox in pd_bboxes])
        dists = np.empty((0, 0))
        if len(gt_bboxes) > 0 and len(pd_bboxes) > 0:
            dists = np.zeros((len(gt_bboxes), len(pd_bboxes)))
    
            for _i in range(len(gt_bboxes)):
                for _j in range(len(pd_bboxes)):
                    _, dists[_i, _j] = iou3d(gt_bboxes[_i][1], pd_bboxes[_j][1])
            dists = np.where(dists >= distth, dists, np.nan)
            dist = -dists

        acc.update(oids, hids, dists, frameid=frame_id)
        names.append(frame_id)
    return acc


def eval_sequence(gt, pd, seq_name, dist='iou', distth=0.5):
    acc = eval_sequence_core(gt, pd, dist=dist, distth=distth)
    return acc