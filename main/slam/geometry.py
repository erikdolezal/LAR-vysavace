import numpy as np


def global_to_local(points, pose) -> np.ndarray:
    """Convert points from global to local coordinates
    From standard (x,y) to (x',y') where x' is forward and y' is left

    Args:
        points (np.ndarray): Points to be converted
        pose (np.ndarray): Position of the car in global coordinates (x,y,yaw)

    Returns:
        np.ndarray: Converted points
    """
    pos = pose[:2]
    ori = pose[2]
    R = np.array([[np.cos(ori), -np.sin(ori)],
                  [np.sin(ori), np.cos(ori)]])
    points[:, :2] -= pos
    points[:, :2] = points[:, :2] @ R
    return points


def local_to_global(points, pose) -> np.ndarray:
    """Convert points from local to global coordinates
    From (x',y') where x' is forward and y' is left to standard (x,y)

    Args:
        points (np.ndarray): Points to be converted
        pose (np.ndarray): Position of the car in global coordinates (x,y,yaw)

    Returns:
        np.ndarray: Converted points
    """
    pos = pose[:2]
    ori = pose[2]
    R_T = np.array([[np.cos(ori), np.sin(ori)],
                    [-np.sin(ori), np.cos(ori)]])

    points[:, :2] = points[:, :2] @ R_T
    points[:, :2] += pos
    return points


def rotate_points(points, angle) -> np.ndarray:
    """Rotate points around the origin

    Args:
        points (np.ndarray): Points to be rotated
        angle (float): Angle in radians

    Returns:
        np.ndarray: Rotated points
    """
    R_T = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    points = points @ R_T
    return points
