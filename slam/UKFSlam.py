import numpy as np
from enum import IntEnum
from scipy.linalg import block_diag
from geometry import global_to_local, local_to_global, rotate_points
from scipy.spatial import cdist

class DataClasses(IntEnum):
    """
    Enum for data classes
    """
    GREEN = 0
    RED = 1
    BLUE = 2
    BALL = 3

config = {
    "pairing_distance" : 0.5
}


class UKF_SLAM():
    """
    Unscented Kalman Filter for estimating speed of the car
    """
    def __init__(self, x_size :int, alpha :float, beta :float, kappa :float):
        self.x = np.zeros(x_size)
        self.P = np.eye(x_size)
        self.data_cls = np.zeros((0, 1))
        self.landmarks = np.zeros((0, 3)) # x, y, class
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n = x_size
        # compute weights for sigma points
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        # mean weights
        self.Wm = np.full(2*self.n + 1, 1/(2*(self.n + self.lambda_))) 
        self.Wm[0] = self.lambda_/(self.n + self.lambda_)
        # covariance weights
        self.Wc = np.full(2*self.n + 1, 1/(2*(self.n + self.lambda_)))
        self.Wc[0] = self.lambda_/(self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)

    def sigma_points(self, x, P):
        """
        Compute sigma points
        x : mean
        P : covariance
        """
        # square root of (self.n + self.lambda_)*P
        U = np.linalg.cholesky((self.n + self.lambda_)*P).T
        # sigma points
        sigmas = np.zeros((2*self.n + 1, self.n))
        sigmas[0] = x.flatten()
        sigmas[1:self.n+1] = x + U
        sigmas[self.n+1:] = x - U

        return sigmas
    
    def f(self, x, u):
        """
        State transition function
        x : state
        u : control input
        """
        x[:,:3] += u
        return x

    def predict(self, u):
        """
        Predict step
        u : odometry
        """
        # compute sigma points
        sigmas = self.sigma_points(self.x, self.P)

        # pass sigma points through state transition function
        self.sigmas_f = self.f(sigmas, u)
        R = np.array([[np.cos(self.x[2]), np.sin(self.x[2])],
                      [-np.sin(self.x[2]), np.cos(self.x[2])]])
        # compute unscented transform
        # mean
        self.x = np.dot(self.Wm, self.sigmas_f)

        # reziduals from mean
        y = self.sigmas_f[1:] - self.x # for n > 0
        y0 = np.expand_dims(self.sigmas_f[0] - self.x, axis=0) # for n = 0
        # change in covariance matrix due to transformation
        cov_difference = self.Wc[1] * y.T @ y + self.Wc[0] * y0.T @ y0 
        # rewrite covariance matrix
        self.P = cov_difference
        self.P = cov_difference.T
        # add process noise
        F = block_diag(R, 1, np.zeros((self.n - 3, self.n - 3)))
        Qc = np.diag(np.hstack(([0.1, 0.1, 0.1], np.zeros(self.n - 3))))
        Q = F @ Qc @ F.T
        self.P += Q
    
    def update(self, z, h, R):
        """
        Update step
        z : measurement
        h : measurement function - returns in the same shape as z
        R : measurement noise covariance
        """
        # compute sigma points
        sigmas = self.sigmas_f # use sigma points from predict step
        
        # pass sigma points through measurement function
        sigmas_h = h(sigmas)

        # compute mean and covariance of transformed sigma points
        z_mean = np.dot(self.Wm, sigmas_h)

        # compute measurement covariance
        z_diff = sigmas_h[1:] - z_mean
        z_diff_0 = np.expand_dims(sigmas_h[0] - z_mean, axis=0)
        Pzz = self.Wc[1] * z_diff.T @ z_diff + self.Wc[0] * z_diff_0.T @ z_diff_0 + R

        # compute cross covariance
        x_diff = sigmas[1:] - self.x
        x_diff_0 = np.expand_dims(sigmas[0] - self.x, axis=0)
        Pxz = self.Wc[1] * x_diff.T @ z_diff + self.Wc[0] * x_diff_0.T @ z_diff_0

        # compute Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # update state and covariance
        self.x += K @ (z - z_mean)
        self.P -= K @ Pzz @ K.T

        return self.x, self.P
    
    def detections_to_lidar(self, detections):
        """
        Convert detections to lidar measurements
        input: 
        detections : array x, y coordinates
        output:
        z : array of measurements dist, angle
        """

        z = np.zeros((len(detections), 2))
        z[:,0] = np.linalg.norm(detections, axis=1)
        z[:,1] = np.arctan2(detections[:,1], detections[:,0])
        return z.flatten()
    
    def h_detections(self, x, matched_detections):
        """
        Measurement function for detections
        x : state
        """
        out = np.zeros((0,matched_detections.shape[0]))
        for sigma in x:
            local_detections = global_to_local(sigma[3:].reshape(-1,2), sigma[:3])[matched_detections]
            out = np.vstack((out, local_detections))
        return out
        
    
    def data_association(self, percep_data):
        """
        Perform data association
        """
        local_landmarks = np.hstack((global_to_local(self.x[3:].reshape(-1,2), self.x[:3]), self.data_cls))
        diff_matrix = np.array([[1,0,0],[0,1,0],[0,0,1e6]])
        dist_mat = cdist(percep_data@diff_matrix.T, local_landmarks[:,:3] @ diff_matrix.T)
        closest_cones = np.empty((0))
        percep_data_mask = np.zeros((percep_data.shape[0],), dtype=bool)
        if local_landmarks.shape[0] > 0 and percep_data.shape[0] > 0: 
            closest_cones = np.argmin(dist_mat, axis=1)
            percep_data_mask = dist_mat[np.arange(0, percep_data.shape[0]), closest_cones] < config["pairing_distance"]
            percep_data_mask &= dist_mat[np.arange(0, percep_data.shape[0]), closest_cones] == np.min(dist_mat.T[closest_cones], axis=1)
        
        return closest_cones, percep_data_mask

    
    def update_from_detections(self, percep_data):
        closest_cones, percep_data_mask = self.data_association(percep_data)
        percep_data_cls = percep_data[:,2]
        lidar_percep_data = self.detections_to_lidar(percep_data[percep_data_mask, :2])
        matched_detections = closest_cones[percep_data_mask]
        def h(x):
            out = np.zeros((0,matched_detections.shape[0]))
            for sigma in x:
                local_detections = global_to_local(sigma[3:].reshape(-1,2), sigma[:3])[matched_detections]
                out = np.vstack((out, local_detections))
            return out




        self.landmarks = np.hstack((self.x[3:].reshape(-1,2), self.data_cls))   

