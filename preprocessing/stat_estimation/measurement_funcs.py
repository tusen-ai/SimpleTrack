import numpy as np, mot_3d.utils as utils
from mot_3d.data_protos import BBox
from scipy.optimize import linear_sum_assignment


def measurement_stats(det_bboxes, gt_bboxes, gt_ids):
    """ process the sequence of detections, gt bboxes, gt ids
        return an array of difference [x, y, z, yaw, l, w, h, x', y', z', yaw']
    """
    # store all the bboxes in the sequence
    seq_map = dict()
    result_diff_val, result_diff_vel_val = list(), list()

    # enumerate all the frames
    frame_num = len(gt_bboxes)
    for frame_index in range(frame_num):
        frame_dets, frame_gts, frame_ids = det_bboxes[frame_index], gt_bboxes[frame_index], gt_ids[frame_index]
        
        if len(frame_gts) == 0 or len(frame_dets) == 0:
            continue

        # pair the detections with gts
        iou_matrix = np.zeros((len(frame_dets), len(frame_gts)))
        for d, det in enumerate(frame_dets):
            for g, gt in enumerate(frame_gts):
                iou_matrix[d, g] = utils.iou3d(det, gt)[1]
        iou_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(iou_matrix)
        matched_indices = np.stack([row_ind, col_ind], axis=1)

        # enumerate all the pairs and get the stats
        pair_num = matched_indices.shape[0]
        for pair_index in range(pair_num):
            det_index, gt_index = matched_indices[pair_index]
            if iou_matrix[det_index][gt_index] > 0.9:
                continue
            else:
                det = frame_dets[det_index]
                gt = frame_gts[gt_index]
                gt_id = frame_ids[gt_index]
                diff_val = BBox.bbox2array(det)[:7] - BBox.bbox2array(gt)[:7]

                if gt_id not in seq_map.keys():
                    seq_map[gt_id] = {frame_index: diff_val}
                else:
                    seq_map[gt_id][frame_index] = diff_val
                result_diff_val += [diff_val[np.newaxis, :]]
                
                # compute the velocity error
                if (frame_index - 1) in seq_map[gt_id].keys():
                    diff_vel_val = diff_val - seq_map[gt_id][frame_index - 1]
                    result_diff_vel_val += [diff_vel_val[np.newaxis, :]]

    if len(result_diff_val) > 0:
        result_diff_val = np.vstack(result_diff_val)
    else:
        result_diff_val = np.empty((0, 7))
    
    if len(result_diff_vel_val) > 0:
        result_diff_vel_val = np.vstack(result_diff_vel_val)
    else:
        result_diff_vel_val = np.empty((0, 7))
    return result_diff_val, result_diff_vel_val