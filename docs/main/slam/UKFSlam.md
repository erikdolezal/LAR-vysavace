Module main.slam.UKFSlam
========================

Classes
-------

`DataClasses(*args, **kwds)`
:   Enum for data classes

    ### Ancestors (in MRO)

    * enum.IntEnum
    * builtins.int
    * enum.ReprEnum
    * enum.Enum

    ### Class variables

    `BALL`
    :

    `BLUE`
    :

    `GREEN`
    :

    `RED`
    :

`UKF_SLAM(x_size: int, alpha: float, beta: float, kappa: float)`
:   Unscented Kalman Filter for estimating speed of the car

    ### Methods

    `cartesian_to_polar_variance(self, x, y, var_x, var_y)`
    :

    `data_association(self, percep_data)`
    :   Perform data association

    `detections_to_lidar(self, detections)`
    :   Convert detections to lidar measurements
        input:
        detections : array x, y coordinates
        output:
        z : array of measurements dist, angle

    `f(self, x, u)`
    :   State transition function
        x : state
        u : control input

    `predict(self, u, old_u)`
    :   Predict step
        u : odometry
        old_u : previous odometry

    `sigma_points(self, x, P)`
    :   Compute sigma points
        x : mean
        P : covariance

    `update(self, z, h, R)`
    :   Update step
        z : measurement
        h : measurement function - returns in the same shape as z
        R : measurement noise covariance

    `update_from_detections(self, percep_data, time)`
    :