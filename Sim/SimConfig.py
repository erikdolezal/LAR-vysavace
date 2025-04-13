from enum import IntEnum
import numpy as np

class SimParm():

    # Visualisation colors 
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # Playground parameters
    WIDTH = 1000
    HEIGHT = 1000
    SIDE_REAL = 5
    SIM_OFFSET = 2.5

    #Simulator paramteres
    SIM_FPS = 100
    SIM_FRICTION = 0.98

    #Sim setup parameters
    GATE_Y_START = 1.5
    MAX_GATE_WIDTH = 0.65
    BALL_GATE_CENTER = 0.3
    MIN_DIS_START = 0.6
    MAX_DIS_START = 2
    CAMERA_FOV = np.deg2rad(60)
    FOV_LINE_LENGTH = 2000

    #Objects parameters
    TUBE_RADIUS = 0.05
    TURTLE_RADIUS = 0.2
    BALL_RADIUS = 0.22

class DataClasses(IntEnum):
    """
    Enum for data classes
    """
    GREEN = 0
    RED = 1
    BLUE = 2
    BALL = 3
    TURTLE = 4
    POINT = 5
