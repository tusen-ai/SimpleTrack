import numpy as np
from ..data_protos import BBox
from filterpy.kalman import KalmanFilter
from .covariance import NuCovariance, WaymoCovariance


class FrameBasedKalmanFilterMotionModel:
    def __init__(self, bbox: BBox, inst_type, covariance='default'):
        # define constant velocity model
        self.score = bbox.s
        self.inst_type = inst_type

        self.kf = KalmanFilter(dim_x=10, dim_z=7) 
        self.kf.x[:7] = BBox.bbox2array(bbox)[:7].reshape((7, 1))
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
                              [0,0,0,0,0,0,1,0,0,0]])

        # # with angular velocity
        # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
        # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
        #                       [0,1,0,0,0,0,0,0,1,0,0],
        #                       [0,0,1,0,0,0,0,0,0,1,0],
        #                       [0,0,0,1,0,0,0,0,0,0,1],  
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0],
        #                       [0,0,0,0,0,0,0,1,0,0,0],
        #                       [0,0,0,0,0,0,0,0,1,0,0],
        #                       [0,0,0,0,0,0,0,0,0,1,0],
        #                       [0,0,0,0,0,0,0,0,0,0,1]])     

        # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
        #                       [0,1,0,0,0,0,0,0,0,0,0],
        #                       [0,0,1,0,0,0,0,0,0,0,0],
        #                       [0,0,0,1,0,0,0,0,0,0,0],
        #                       [0,0,0,0,1,0,0,0,0,0,0],
        #                       [0,0,0,0,0,1,0,0,0,0,0],
        #                       [0,0,0,0,0,0,1,0,0,0,0]])

        self.covariance_type = covariance
        if self.covariance_type == 'default':
            # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
            self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
            self.kf.P *= 10.
    
            # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
            self.kf.Q[7:, 7:] *= 0.01
        elif 'nuscenes' in self.covariance_type:
            cov_name = self.covariance_type.split('_')[1]
            cov = NuCovariance(cov_name)
            self.kf.P = cov.P[inst_type][:-1, :-1]
            self.kf.Q = cov.Q[inst_type][:-1, :-1]
            self.kf.R = cov.R[inst_type][:, :]
        elif 'waymo' in self.covariance_type:
            cov_name = self.covariance_type.split('_')[1]
            cov = WaymoCovariance(cov_name)

        self.history = [bbox]

    def update(self, det_bbox: BBox, aux_info): 
        """ 
        Updates the state vector with observed bbox.
        """
        bbox = BBox.bbox2array(det_bbox)[:7]

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

        self.kf.update(bbox)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        if det_bbox.s is None:
            self.score = self.score * 0.1
        else:
            self.score = det_bbox.s
        
        cur_bbox = self.kf.x[:7].reshape(-1).tolist()
        cur_bbox = BBox.array2bbox(cur_bbox + [self.score])
        self.history[-1] = cur_bbox
        return
    
    def get_prediction(self, time_stamp):
        return self.predict()

    def predict(self):       
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()      
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        pred_bbox = BBox.array2bbox(self.kf.x[:7].reshape(-1))
        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.history[-1]
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
    
    def sync_time_stamp(self, time_stamp):
        pass