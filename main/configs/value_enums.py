from enum import IntEnum


class DataClasses(IntEnum):
    """
    Enum for data classes
    """

    GREEN = 0
    RED = 1
    BLUE = 2
    BALL = 3


class ErrorCodes(IntEnum):
    """
    Enum for data classed.
    """
    OK_ERR = 0
    MORE_BLUE_ERR = 1
    NO_BALL_ERR = 2
    BALL_STUCK_ERR = 3
    NO_SHOOT_ERR = 4
    NO_ROBOT_ERR = 5
    ZERO_BLUE_ERR = 6
    IS_GOAL_ERR = 7
