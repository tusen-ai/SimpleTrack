""" naive motion model:
    1. store the bbox of each time stamp
    2. make prediction according to the previous velocity
"""
import numpy as np
from mot_3d.data_protos import BBox
from copy import deepcopy


class NaiveMotionModel:
    def __init__(self, bbox: BBox, velo, inst_type, time_stamp):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.time_stamp = time_stamp
        self.score = bbox.s
        self.inst_type = inst_type

        self.history = [bbox]
        self.velos = [velo]
    
    def get_prediction(self, time_stamp):
        """
        On unassociated cases, use prediction
        """
        prev_bbox = self.history[-1]
        velo = self.velos[-1]
        result_bbox = BBox()
        BBox.copy_bbox(result_bbox, prev_bbox)

        time_lag = time_stamp - self.prev_time_stamp
        result_bbox.x += velo[0] * time_lag
        result_bbox.y += velo[1] * time_lag
        result_bbox.s = self.score * 0.01
        self.history.append(result_bbox)
        self.velos.append(self.velos[-1])
        return result_bbox
    
    def update(self, det_bbox: BBox, aux_info):
        """
        incoming detections for update
        """
        velo = aux_info['velo']
        self.history[-1] = det_bbox
        self.velos[-1] = velo
        self.score = det_bbox.s
        return
    
    def get_state(self):
        cur_bbox = self.history[-1]
        return cur_bbox
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        self.prev_time_stamp = time_stamp
        return
