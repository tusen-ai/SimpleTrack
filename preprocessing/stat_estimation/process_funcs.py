import numpy as np
from mot_3d.data_protos import BBox


def process_stats(gt_bboxes, gt_ids):
    """ process the sequence of gt bboxes and gt ids
        return an array of difference [x, y, z, yaw, l, w, h]
    """
    # store all the bboxes in the sequence
    seq_map = dict()

    # enumerate over all the bboxes
    # compute their information, including location, velocity, and acceleration
    frame_num = len(gt_bboxes)
    for frame_index in range(frame_num):
        frame_bboxes, frame_ids = gt_bboxes[frame_index], gt_ids[frame_index]
        for i, (bbox, id) in enumerate(zip(frame_bboxes, frame_ids)):
            bbox_data = np.asarray([
                bbox.x, bbox.y, bbox.z, bbox.o,
                bbox.l, bbox.w, bbox.h,
                0, 0, 0, 0,
                0, 0, 0, 0
            ])

            # store the instance in the sequence map
            if id not in seq_map.keys():
                seq_map[id] = {frame_index: bbox_data}
            else:
                seq_map[id][frame_index] = bbox_data
            
            # if we can find the same object in the previous frame, get the velocity
            if (frame_index - 1) in seq_map[id].keys():
                velo = bbox_data[:4] - seq_map[id][frame_index - 1][:4]
                seq_map[id][frame_index][7:11] = velo

                # back fill to the first element
                if len(seq_map[id]) == 2:
                    seq_map[id][frame_index - 1][7:11] = velo
                
                # if we can find the same object in the previous 2 frames, get the acceleration
                if (frame_index - 2) in seq_map[id].keys():
                    prev_velo = seq_map[id][frame_index - 1][:4] - seq_map[id][frame_index - 2][:4]
                    accel = velo - prev_velo
                    seq_map[id][frame_index][11:] = accel
                    
                    # back fill
                    if len(seq_map[id]) == 3:
                        seq_map[id][frame_index - 1][11:] = accel
                        seq_map[id][frame_index - 2][11:] = accel
    
    # aggregate all the bbox data
    result_data = list()
    for id in seq_map.keys():
        for frame_key in seq_map[id].keys():
            result_data.append(seq_map[id][frame_key][np.newaxis, :])
    if len(result_data) > 0:
        result_data = np.vstack(result_data)
    else:
        result_data = np.empty((0, 15))
    
    return result_data