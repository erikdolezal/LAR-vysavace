from configs.value_enums import DataClasses


slam_config = {
    "pairing_distance": 0.6,
    "detection_var": 0.2,
    "position_var": 0.02,
    "rotation_var": 0.001,
    "min_occurences": 2,
    "detection_timeout": 1,
}

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

vision_config = {
    "show": False,
    "cls_to_col": {DataClasses.GREEN: (0, 255, 0), 
                   DataClasses.RED: (0, 0, 255), 
                   DataClasses.BLUE: (255, 0, 0), 
                   DataClasses.BALL: (0, 255, 255)},
    "class_map": {0: DataClasses.BALL, 
                   1: DataClasses.BALL, 
                   2: DataClasses.BLUE, 
                   3: DataClasses.GREEN,
                   4: DataClasses.RED
    }
}

velocity_control_config = {
    "max_acc": 0.8,  # m/s^2
    "max_speed": 0.8,  # m/s
    "max_ang_speed": 0.8,  # rad/s
    "ang_p": 1.5,
}