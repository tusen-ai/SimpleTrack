""" a finite state machine to manage the life cycle
    states: 
      - birth: first founded
      - alive: alive
      - no_asso: without high score association, about to die
      - dead: may it eternal peace
"""
import numpy as np
from ..data_protos import Validity
from ..update_info_data import UpdateInfoData
from .. import utils


class HitManager:
    def __init__(self, configs, frame_index):
        self.time_since_update = 0
        self.hits = 1           # number of total hits including the first detection
        self.hit_streak = 1     # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.recent_state = None

        self.max_age = configs['running']['max_age_since_update']
        self.min_hits = configs['running']['min_hits_to_birth']

        self.state = 'birth'
        self.recent_state = 1
        self.no_asso = False
        if frame_index <= self.min_hits or self.min_hits == 0:
            self.state = 'alive'
            self.recent_state = 1
    
    def predict(self, is_key_frame):
        # only on key frame
        # unassociated prediction affects the life-cycle management
        if not is_key_frame:
            return

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1
        self.fall = True
        return
    
    def if_valid(self, update_info):
        self.recent_state = update_info.mode
        return update_info.mode
    
    def update(self, update_info: UpdateInfoData, is_key_frame=True):
        # the update happening during the non-key-frame
        # can extend the life of tracklet
        association = self.if_valid(update_info)
        self.recent_state = association
        if association != 0:
            self.fall = False
            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1 # number of continuing hit
            if self.still_first:
                self.first_continuing_hit += 1 # number of continuing hit in the fist time
        if is_key_frame:
            self.state_transition(association, update_info.frame_index)
    
    def state_transition(self, mode, frame_index):
        # if just founded
        if self.state == 'birth':
            if (self.hits >= self.min_hits) or (frame_index <= self.min_hits):
                self.state = 'alive'
                self.recent_state = mode
            elif self.time_since_update >= self.max_age:
                self.state = 'dead'
        # already alive
        elif self.state == 'alive':
            if self.time_since_update >= self.max_age:
                self.state = 'dead'
        
    def alive(self, frame_index):
        return self.state == 'alive'
    
    def death(self, frame_index):
        return self.state == 'dead'
    
    def valid_output(self, frame_index):
        return (self.state == 'alive') and (self.no_asso == False)
    
    def state_string(self, frame_index):
        if self.state == 'birth':
            return '{:}_{:}'.format(self.state, self.hits)
        elif self.state == 'alive':
            return '{:}_{:}_{:}'.format(self.state, self.recent_state, self.time_since_update)
        elif self.state == 'dead':
            return '{:}_{:}'.format(self.state, self.time_since_update)