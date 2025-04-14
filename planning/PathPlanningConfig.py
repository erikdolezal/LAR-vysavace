from enum import IntEnum

class PlanningParm():
    CLEARANCE = 0.5
    SHOOT_SCALING = 2
    SHOOT_STEPBACK = 0.6
    BALL_PROXIMITY = 0.2
    HADING_CHECK = 0.0873*2
    ROBOT_TURN_RADIUS = 0.3
    GOAL_POX = 0.7
    SHOOT_ALIGNMENT = 0.1
    GOAL_CHECK = 0.1
        
class DataClasses(IntEnum):
    """
    Enum for data classes
    """
    GREEN = 0
    RED = 1
    BLUE = 2
    BALL = 3


class ErrorCodes(IntEnum):
    OK_ERR = 0
    MORE_BLUE_ERR = 1
    NO_BALL_ERR = 2
    BALL_STUCK_ERR = 3
    NO_SHOOT_ERR = 4
    NO_ROBOT_ERR = 5
    ZERO_BLUE_ERR = 6
    IS_GOAL_ERR = 7