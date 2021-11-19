# Selecting the sequences according to types
# Transfer the ID from string into int if needed
from ..data_protos import BBox
import numpy as np


__all__ = ['inst_filter', 'str2int', 'box_wrapper', 'type_filter', 'id_transform']


def str2int(strs):
    result = [int(s) for s in strs]
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


def id_transform(ids):
    frame_num = len(ids)

    id_list = list()
    for _i in range(frame_num):
        id_list += ids[_i]
    id_list = sorted(list(set(id_list)))
    
    id_mapping = dict()
    for _i, id in enumerate(id_list):
        id_mapping[id] = _i
    
    result = list()
    for _i in range(frame_num):
        frame_ids = list()
        frame_id_num = len(ids[_i])
        for _j in range(frame_id_num):
            frame_ids.append(id_mapping[ids[_i][_j]])
        result.append(frame_ids)
    return result    


def inst_filter(ids, bboxes, types, type_field=[1], id_trans=False):
    """ filter the bboxes according to types
    """
    frame_num = len(ids)
    if id_trans:
        ids = id_transform(ids)
    id_result, bbox_result = [], []
    for _i in range(frame_num):
        frame_ids = list()
        frame_bboxes = list()
        frame_id_num = len(ids[_i])
        for _j in range(frame_id_num):
            obj_type = types[_i][_j]
            matched = False
            for type_name in type_field:
                if str(type_name) in str(obj_type):
                    matched = True
            if matched:
                frame_ids.append(ids[_i][_j])
                frame_bboxes.append(BBox.array2bbox(bboxes[_i][_j]))
        id_result.append(frame_ids)
        bbox_result.append(frame_bboxes)
    return id_result, bbox_result


def type_filter(contents, types, type_field=[1]):
    frame_num = len(types)
    content_result = [list() for i in range(len(type_field))]
    for _k, inst_type in enumerate(type_field):
        for _i in range(frame_num):
            frame_contents = list()
            frame_id_num = len(contents[_i])
            for _j in range(frame_id_num):
                if types[_i][_j] != inst_type:
                    continue
                frame_contents.append(contents[_i][_j])
            content_result[_k].append(frame_contents)
    return content_result