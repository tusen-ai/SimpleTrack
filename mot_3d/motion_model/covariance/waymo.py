import numpy as np, json, os


class WaymoCovariance:
    def __init__(self, name):
        self.name = name
        self.path = 'preprocessing/waymo_data/waymo_stats'

        P = json.load(open(os.path.join(self.path, 'P_{:}.json'.format(name)), 'r'))
        Q = json.load(open(os.path.join(self.path, 'Q_{:}.json'.format(name)), 'r'))
        R = json.load(open(os.path.join(self.path, 'R_{:}.json'.format(name)), 'r'))

        self.obj_types = ['vehicle', 'cyclist', 'pedestrian']
        self.P = {obj_type: np.diag(P[obj_type]) for obj_type in self.obj_types}
        self.Q = {obj_type: np.diag(Q[obj_type]) for obj_type in self.obj_types}
        self.R = {obj_type: np.diag(R[obj_type]) for obj_type in self.obj_types}
        return
    