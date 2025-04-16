import numpy as np
from scipy.linalg import block_diag
from algorithms.geometry import global_to_local, local_to_global, rotate_points
from scipy.spatial.distance import cdist
from configs.alg_config import slam_config as config
from configs.value_enums import DataClasses


class UKF_SLAM:
    """
    Unscented Kalman Filter SLAM for estimatin robots position nad mapping landmarks
    Attributes:
        x (np.ndarray): State vector of the robot and landmarks.
        P (np.ndarray): Covariance matrix of the state.
        data_cls (np.ndarray): Array storing the classes of detected landmarks.
        landmarks (np.ndarray): Array of landmarks with their positions and classes.
        alpha (float): Scaling parameter for sigma points.
        beta (float): Parameter for incorporating prior knowledge of the distribution.
        kappa (float): Secondary scaling parameter for sigma points.
        n (int): Dimensionality of the state vector.
        lambda_ (float): Composite scaling parameter for sigma points.
        Wm (np.ndarray): Weights for the mean of sigma points.
        Wc (np.ndarray): Weights for the covariance of sigma points.
        landmark_contestants (np.ndarray): Temporary storage for potential landmarks.
    Methods:
        __init__(x_size, alpha, beta, kappa):
            Initializes the UKF_SLAM object with state size and filter parameters.
        sigma_points(x, P):
            Computes sigma points for the given state mean and covariance.
        f(x, u):
            State transition function for predicting the next state.
        predict(u, old_u):
            Performs the prediction step of the UKF using odometry data.
        update(z, h, R):
            Performs the update step of the UKF using camera detections.
        data_association(percep_data):
            Associates perceived data with existing landmarks.
        update_from_detections(percep_data, time):
            Updates the state and landmarks based on new detections.
    """

    def __init__(self, x_size: int, alpha: float, beta: float, kappa: float):
        self.x = np.zeros(x_size)
        self.P = np.eye(x_size) * 0.5
        self.data_cls = np.zeros((0))
        self.landmarks = np.zeros((0, 3))  # x, y, class
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n = x_size
        # compute weights for sigma points
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        # mean weights
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        # covariance weights
        self.Wc = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        self.landmark_contestants = np.zeros((0, 5))  # x, y, cls, seen, last_occurence

    def sigma_points(self, x, P):
        """
        Compute sigma points for calculating Unscented transform.
        Args:
            x (np.ndarray): mean
            P (np.ndarray): covariance
        """
        # square root of (self.n + self.lambda_)*P
        U = np.linalg.cholesky((self.n + self.lambda_) * P).T
        # sigma points
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x.flatten()
        sigmas[1: self.n + 1] = x + U
        sigmas[self.n + 1:] = x - U

        return sigmas

    def f(self, x, u):
        """
        State transition function.
        Args:
            x (np.ndarray): state
            u (np.ndarray): control input
        """
        x[:, :3] += u
        return x

    def predict(self, u, old_u):
        """
        Predict step of Kalman filter.
        Args:
            u (np.ndarray): odometry
            old_u (np.ndarray): previous odometry
        """
        # compute delta position and rotation
        delta_theta = np.tan(np.arctan(u[2] - old_u[2]))
        delta_pos = np.hstack((rotate_points(u[:2] - old_u[:2], -u[2]), delta_theta))

        # compute sigma points
        sigmas = self.sigma_points(self.x, self.P)

        # pass sigma points through state transition function
        R = np.array([[np.cos(self.x[2]), np.sin(self.x[2])], [-np.sin(self.x[2]), np.cos(self.x[2])]])
        delta_pos[:2] = rotate_points(delta_pos[:2], self.x[2])
        self.sigmas_f = self.f(sigmas, delta_pos)
        # compute unscented transform
        # mean
        self.x = np.dot(self.Wm, self.sigmas_f)

        # reziduals from mean
        y = self.sigmas_f[1:] - self.x  # for n > 0
        y0 = np.expand_dims(self.sigmas_f[0] - self.x, axis=0)  # for n = 0
        # change in covariance matrix due to transformation
        cov_difference = self.Wc[1] * y.T @ y + self.Wc[0] * y0.T @ y0
        # rewrite covariance matrix
        self.P = cov_difference
        self.P = cov_difference.T
        # add process noise
        F = block_diag(R, 1, np.zeros((self.n - 3, self.n - 3)))
        Qc = np.diag(np.hstack(([config["position_var"]] * 2, [config["rotation_var"]], np.zeros(self.n - 3))))
        Q = F @ Qc @ F.T
        self.P += Q

    def update(self, z, h, R):
        """
        Update step of Kalman Filter.
        Args:
            z (np.ndarray): measurement
            h (function): measurement function - returns in the same shape as z
            R (np.ndarray): measurement noise covariance
        """
        # compute sigma points
        # sigmas = self.sigmas_f # use sigma points from predict step
        sigmas = self.sigma_points(self.x, self.P)

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

    def data_association(self, percep_data):
        """
        Perform data association based on minimal distance between perceived data and landmarks.
        Args:
            percep_data (np.ndarray): perceived data
        Returns:
            tuple: A tuple containing:
                np.ndarray: indices of closest cones
                np.ndarray: mask for perceived data
        """
        # convert landmark to local coordinates
        local_landmarks = np.hstack(
            (global_to_local(self.x[3:].reshape(-1, 2).copy(), self.x[:3]), np.expand_dims(self.data_cls, axis=1))
        )
        # calculate distance matrix of perception data and local landmarks
        diff_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e6]])
        dist_mat = cdist(percep_data.copy() @ diff_matrix.T, local_landmarks[:, :3] @ diff_matrix.T)
        closest_cones = np.empty((0))
        percep_data_mask = np.zeros((percep_data.shape[0],), dtype=bool)
        if local_landmarks.shape[0] > 0 and percep_data.shape[0] > 0:
            # find closest cones
            closest_cones = np.argmin(dist_mat, axis=1)
            # find closest cones that are within the pairing distance
            percep_data_mask = dist_mat[np.arange(0, percep_data.shape[0]), closest_cones] < config["pairing_distance"]
            # remove closest cones that are not the closest to the perception data
            percep_data_mask &= dist_mat[np.arange(0, percep_data.shape[0]), closest_cones] == np.min(
                dist_mat.T[closest_cones], axis=1
            )
        return closest_cones, percep_data_mask

    def update_from_detections(self, percep_data, time):
        """
        Updates the state of the SLAM system based on perception data and create map of landmarks.
        Args:
            percep_data (numpy.ndarray): A 2D array containing perception data.
            time (float): The current timestamp.
        """
        # remove ball from perception data
        percep_data = percep_data[percep_data[:, 2] != DataClasses.BALL]

        if self.landmarks.shape[0] > 0:
            # associate perception data with landmarks
            closest_cones, percep_data_mask = self.data_association(percep_data)
            percep_data_cls = percep_data[:, 2]
            matched_detections = closest_cones[percep_data_mask]
            if matched_detections.shape[0] > 0:
                # define measurement function
                def h(x):
                    out = np.zeros((0, matched_detections.shape[0] * 2))
                    for sigma in x:
                        # convert sigma points to local coordinates
                        local_detections = global_to_local(sigma[3:].copy().reshape(-1, 2), sigma[:3])[
                            matched_detections
                        ]
                        out = np.vstack((out, local_detections.flatten()))
                    return out
                # define measurement noise covariance
                R = np.diag(np.ones((matched_detections.shape[0] * 2)) * config["detection_var"])
                # update state and covariance
                self.update(percep_data[percep_data_mask, :2].flatten(), h, R)
            # landmark contestants
            new_percep_data = local_to_global(percep_data[~percep_data_mask].copy(), self.x[:3])[:, :3]
            new_percep_data_cls = percep_data_cls[~percep_data_mask]
        else:
            # landmark contestants
            new_percep_data = local_to_global(percep_data.copy(), self.x[:3])[:, :3]
            new_percep_data_cls = percep_data[:, 2]

        # calculate distance matrix of perception data and local landmark contestants
        diff_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e6]])
        dist_mat = cdist(new_percep_data.copy() @ diff_matrix.T, self.landmark_contestants[:, :3] @ diff_matrix.T)
        new_data_mask = np.zeros(new_percep_data.shape[0], dtype=bool)
        if self.landmark_contestants.shape[0] > 0 and new_percep_data.shape[0] > 0:
            # associate perception data with landmark contestants
            closest_landmarks = np.argmin(dist_mat, axis=1)
            new_data_mask = (
                dist_mat[np.arange(0, new_percep_data.shape[0]), closest_landmarks] < config["pairing_distance"]
            )
            new_data_mask &= dist_mat[np.arange(0, new_percep_data.shape[0]), closest_landmarks] == np.min(
                dist_mat.T[closest_landmarks], axis=1
            )
            # update landmark contestants position
            self.landmark_contestants[closest_landmarks[new_data_mask], :2] += (
                new_percep_data[new_data_mask, :2] - self.landmark_contestants[closest_landmarks[new_data_mask], :2]
            ) * np.vstack(
                (
                    1 / (self.landmark_contestants[closest_landmarks[new_data_mask], 3] + 1),
                    1 / (self.landmark_contestants[closest_landmarks[new_data_mask], 3] + 1),
                )
            ).T
            # update landmark contestants ocurrences
            self.landmark_contestants[closest_landmarks[new_data_mask], 3] += 1
            # update landmark contestants last occurence time
            self.landmark_contestants[closest_landmarks[new_data_mask], 4] = time

        # add new landmark contestants
        new_contestants = np.hstack(
            (new_percep_data[~new_data_mask], np.ones((np.sum((~new_data_mask)), 2)) * np.array([1, time]))
        )
        self.landmark_contestants = np.vstack((self.landmark_contestants, new_contestants))
        # remove landmark contestants that are too old
        self.landmark_contestants = self.landmark_contestants[
            time - self.landmark_contestants[:, 4] < config["detection_timeout"]
        ]
        # pick landmark contestants that will be added to the state
        new_data_mask = self.landmark_contestants[:, 3] > config["min_occurences"]
        new_percep_data = self.landmark_contestants[new_data_mask, :3]
        new_percep_data_cls = new_percep_data[:, 2]
        self.landmark_contestants = self.landmark_contestants[~new_data_mask]
        print(f"landmark contestants {self.landmark_contestants}")

        # add new landmarks to the state
        self.data_cls = np.hstack((self.data_cls, new_percep_data_cls))
        self.x = np.hstack((self.x, new_percep_data[:, :2].flatten()))
        # add new landmarks to the covariance matrix
        self.P = block_diag(self.P, np.eye(new_percep_data.shape[0] * 2) * 0.5)
        self.landmarks = np.hstack((self.x[3:].reshape(-1, 2), np.expand_dims(self.data_cls, axis=1)))
        # update sigma point weights
        self.kappa = 3 - self.x.shape[0]

        self.n = self.x.shape[0]
        # compute weights for sigma points
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        # mean weights
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        # covariance weights
        self.Wc = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)
