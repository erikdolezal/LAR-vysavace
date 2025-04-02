from enum import IntEnum

class PlanningParm():
    CLEARANCE = 0.4
    SHOOT_SCALING = 10
    SHOOT_STEPBACK = 1
    BALL_PROXIMITY = 0.3
        
class DataClasses(IntEnum):
    """
    Enum for data classes
    """
    GREEN = 0
    RED = 1
    BLUE = 2
    BALL = 3
    TURTLE = 4
