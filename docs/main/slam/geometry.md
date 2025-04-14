Module main.slam.geometry
=========================

Functions
---------

`global_to_local(points, pose) ‑> numpy.ndarray`
:   Convert points from global to local coordinates
    From standard (x,y) to (x',y') where x' is forward and y' is left
    
    Args:
        points (np.ndarray): Points to be converted
        pose (np.ndarray): Position of the car in global coordinates (x,y,yaw)
    
    Returns:
        np.ndarray: Converted points

`local_to_global(points, pose) ‑> numpy.ndarray`
:   Convert points from local to global coordinates
    From (x',y') where x' is forward and y' is left to standard (x,y)
    
    Args:
        points (np.ndarray): Points to be converted
        pose (np.ndarray): Position of the car in global coordinates (x,y,yaw)
    
    Returns:
        np.ndarray: Converted points

`rotate_points(points, angle) ‑> numpy.ndarray`
:   Rotate points around the origin
    
    Args:
        points (np.ndarray): Points to be rotated
        angle (float): Angle in radians
    
    Returns:
        np.ndarray: Rotated points