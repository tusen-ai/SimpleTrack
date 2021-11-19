""" input form of the data in each frame
"""
from .data_protos import BBox
import numpy as np, mot_3d.utils as utils


class FrameData:
    def __init__(self, dets, ego, time_stamp=None, pc=None, det_types=None, aux_info=None):
        self.dets = dets         # detections for each frame
        self.ego = ego           # ego matrix information
        self.pc = pc
        self.det_types = det_types
        self.time_stamp = time_stamp
        self.aux_info = aux_info

        for i, det in enumerate(self.dets):
            self.dets[i] = BBox.array2bbox(det)
        
        # if not aux_info['is_key_frame']:
        #     self.dets = [d for d in self.dets if d.s >= 0.5]