""" naive motion model:
    1. store the bbox of each time stamp
    2. make prediction according to the previous velocity
"""
import numpy as np
from mot_3d.data_protos import BBox
from copy import deepcopy


class NaiveMAMotionModel:
    def __init__(self, bbox: BBox, inst_type, time_stamp):
        # the time stamp of last observation
        self.prev_time_stamp = time_stamp
        self.latest_time_stamp = time_stamp
        self.score = bbox.s
        self.inst_type = inst_type
        self.ma_velo = np.array([0, 0, 0])
        self.history = [bbox]
        self.time_lag = 0
    
    def get_prediction(self, time_stamp):
        """
        On unassociated cases, use prediction
        """
        self.latest_time_stamp = time_stamp
        prev_bbox = self.history[-1]
        result_bbox = BBox()
        BBox.copy_bbox(result_bbox, prev_bbox)

        time_lag = time_stamp - self.prev_time_stamp
        result_bbox.x += self.ma_velo[0]
        result_bbox.y += self.ma_velo[1]
        result_bbox.z += self.ma_velo[2]
        result_bbox.s = self.score * 0.01
        self.history.append(result_bbox)
        return result_bbox
    
    def update(self, det_bbox: BBox, aux_info):
        """
        incoming detections for update
        """
        self.history[-1] = det_bbox
        self.score = det_bbox.s

        # movement = np.array([
        #     self.history[-1].x - self.history[-2].x,
        #     self.history[-1].y - self.history[-2].y
        # ])

        time_lag = self.latest_time_stamp - self.prev_time_stamp

        horizon = min(len(self.history) - 1, 4)
        movement = np.zeros(3)
        for i in range(horizon):
            movement[0] += (self.history[-1 - i].x - self.history[-2 - i].x)
            movement[1] += (self.history[-1 - i].y - self.history[-2 - i].y)
            movement[2] += (self.history[-1 - i].z - self.history[-2 - i].z)
        self.ma_velo = movement / (horizon + 1e-8)
        
        # if len(self.history) == 2:
        #     self.ma_velo = movement
        # else:
        #     self.ma_velo = movement * 0.5 + self.ma_velo * 0.5
        self.prev_time_stamp = self.latest_time_stamp
        return
    
    def get_state(self):
        cur_bbox = self.history[-1]
        return cur_bbox
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return
