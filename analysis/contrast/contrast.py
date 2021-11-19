import numpy as np, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity


__all__ = ['frame_contrast']


def compute_ious(preds, gts):
    if len(preds) == 0:
        return []
    if len(gts) == 0:
        return np.zeros(len(preds))
    
    ious = np.zeros((len(preds), len(gts)))
    for i, pd in enumerate(preds):
        for j, gt in enumerate(gts):
            ious[i, j] = utils.iou3d(pd, gt)[0]
    
    max_ious = np.max(ious, axis=1)
    return max_ious


def tp(iou):
    return 1 if iou >= 0.7 else 0


def frame_contrast(pred_bboxesa, pred_bboxesb, gt_bboxes):
    """ indexes of 
        normal pred_bboxesa, normal pred_bboxesb, unmatched pred_bboxesa, unmatched pred_bboxesb
        iou values
        iousa, iousb
    """
    # 0. compute the max iou values with gt
    iousa = compute_ious(pred_bboxesa, gt_bboxes)
    iousb = compute_ious(pred_bboxesb, gt_bboxes)

    # 1. dealing with extreme cases, either a or b has no bboxes
    if len(pred_bboxesa) == 0:
        return [], [], [], list(range(len(pred_bboxesb))), iousa, iousb
    if len(pred_bboxesb) == 0:
        return [], [], list(range(len(pred_bboxesa))), [], iousa, iousb

    # 2. match the preds into pairs
    match_a, unmatch_a, match_b, unmatch_b = list(), list(), list(), list()
    ious = np.zeros((len(pred_bboxesa), len(pred_bboxesb)))
    for i, pda in enumerate(pred_bboxesa):
        for j, pdb in enumerate(pred_bboxesb):
            ious[i, j] = utils.iou3d(pda, pdb)[1]
    max_iou_index = np.argmax(ious, axis=1)
    
    for i, _ in enumerate(pred_bboxesa):
        if ious[i, max_iou_index[i]] >= 0.1:
            match_a += [i]
            match_b += [max_iou_index[i]]
        else:
            unmatch_a += [i]
    
    for i, _ in enumerate(pred_bboxesb):
        if i not in match_b:
            unmatch_b += [i]   

    # 3. check the pairs
    normal_a, strange_a, normal_b, strange_b = list(), list(), list(), list()
    for i, j in zip(match_a, match_b):
        if tp(iousa[i]) == tp(iousb[j]):
            normal_a += [i]
            normal_b += [j]
        else:
            strange_a += [i]
            strange_b += [j]
    strange_a += unmatch_a
    strange_b += unmatch_b

    # 4. return the results
    return normal_a, normal_b, strange_a, strange_b, iousa, iousb