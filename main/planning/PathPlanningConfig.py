from enum import IntEnum


class PlanningParm():
    """
    PlanningParm is a configuration class.
    Attributes:
        CLEARANCE (float): Minimum clearance distance required around obstacles.
        SHOOT_SCALING (float): Scaling factor for shooting calculations.
        SHOOT_STEPBACK (float): Distance to step back before kicking the ball.
        BALL_PROXIMITY (float): Proximity threshold to the ball.
        HADING_CHECK (float): Heading check threshold in radians.
        ROBOT_TURN_RADIUS (float): Turning radius of the robot.
        GOAL_POX (float): Proximity threshold to the goal position.
        SHOOT_ALIGNMENT (float): Alignment threshold for shooting.
        GOAL_CHECK (float): Position of checks points for goal. Behind and in front of the goal line.
        ROBOT_DIST_MIN (float): How close to the goal line, can the robot get to.
    """
    CLEARANCE = 0.5
    SHOOT_SCALING = 2
    SHOOT_STEPBACK = 0.8
    BALL_PROXIMITY = 0.2
    HADING_CHECK = 0.0873 * 2
    ROBOT_TURN_RADIUS = 0.3
    GOAL_POX = 0.7
    SHOOT_ALIGNMENT = 0.1
    GOAL_CHECK = 0.1
    ROBOT_DIST_MIN = 0.48


class DataClasses(IntEnum):
    """
    Enum for data classes.
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
