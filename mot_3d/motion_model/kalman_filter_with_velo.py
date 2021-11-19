import numpy as np
from mot_3d.data_protos import BBox
from filterpy.kalman import KalmanFilter
from mot_3d.motion_model.covariance import NuCovariance, WaymoCovariance


class KalmanFilterVeloMotionModel:
    def __init__(self, bbox: BBox, velo, inst_type, time_stamp, covariance='default'):
        # define constant velocity model
        self.time_stamp = time_stamp
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=9) 
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))  # bbox
        self.kf.x[7:9] = np.asarray(velo).reshape((2, 1))  # velocity
        self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                              [0,1,0,0,0,0,0,0,1,0],
                              [0,0,1,0,0,0,0,0,0,1],
                              [0,0,0,1,0,0,0,0,0,0],  
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                              [0,1,0,0,0,0,0,0,0,0],
                              [0,0,1,0,0,0,0,0,0,0],
                              [0,0,0,1,0,0,0,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,1,0,0,0,0],
                              [0,0,0,0,0,0,1,0,0,0],
                              [0,0,0,0,0,0,0,1,0,0],
                              [0,0,0,0,0,0,0,0,1,0]])

        self.covariance_type = covariance
        if self.covariance_type == 'default':
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            # self.kf.P[7:, 7:] *= 1000.
            self.kf.P *= 10.
            # self.kf.Q[7:, 7:] *= 0.01

        self.history = []

    def update(self, det_bbox: BBox, aux_info): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7].tolist()
        velo = aux_info['velo']
        ######################### orientation correction
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        new_theta = bbox[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox[3] = new_theta

        predicted_theta = self.kf.x[3]
        if np.abs(new_theta - predicted_theta) > np.pi / 2.0 and np.abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi       
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if np.abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0: self.kf.x[3] += np.pi * 2
            else: self.kf.x[3] -= np.pi * 2

        #########################     # flip

        self.kf.update(np.asarray(bbox + velo))

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            self.score = self.score * 0.1
        else:
            self.score = det_bbox.s
        return

    def predict(self, time_stamp=None):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        time_lag = time_stamp - self.time_stamp
        self.kf.F[7:9, 7:9] = np.eye(2) * time_lag

        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.history.append(self.kf.x)
        bbox = self.history[-1].reshape(-1)
        return BBox.array2bbox(bbox)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        return BBox.array2bbox(cur_bbox + [self.score])
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp
        return