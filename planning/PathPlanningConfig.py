from enum import IntEnum

class PlanningParm():
    CLEARANCE = 0.4
    SHOOT_SCALING = 10
    SHOOT_STEPBACK = 0.5
    BALL_PROXIMITY = 0.3
    HADING_CHECK = 0.0873*2
    ROBOT_TURN_RADIUS = 0.3
        
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
    NO_RORBOT_ERR = 5
    ZERO_BLUE_ERR = 6