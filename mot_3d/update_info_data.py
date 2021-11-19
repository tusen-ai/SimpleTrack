""" a general interface for aranging the things inside a single tracklet
"""
from .data_protos import BBox
from . import utils
import numpy as np


class UpdateInfoData:
    def __init__(self, mode, bbox, frame_index, ego, dets=None, pc=None, aux_info=None):
        self.mode = mode
        self.bbox = bbox
        self.ego = ego
        self.frame_index = frame_index
        self.pc = pc
        self.dets = dets
        self.aux_info = aux_info
