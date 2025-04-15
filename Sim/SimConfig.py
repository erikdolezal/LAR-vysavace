from enum import IntEnum
import numpy as np

class SimParm():
    """
    SimParm is a configuration class that defines various parameters and constants 
    used in the simulation. These parameters include visual settings, playground 
    dimensions, simulator settings, setup configurations, and object properties.
    Attributes:
        WHITE (tuple): RGB color for white.
        BLACK (tuple): RGB color for black.
        RED (tuple): RGB color for red.
        YELLOW (tuple): RGB color for yellow.
        GREEN (tuple): RGB color for green.
        BLUE (tuple): RGB color for blue.
        WIDTH (int): Width of the simulation playground in pixels.
        HEIGHT (int): Height of the simulation playground in pixels.
        SIDE_REAL (float): Real-world side length of the playground in meters.
        SIM_OFFSET (float): Offset for simulation scaling.
        # Simulator parameters
        SIM_FPS (int): Frames per second for the simulation.
        SIM_FRICTION (float): Friction coefficient in the simulation.
        # Sim setup parameters
        GATE_Y_START (float): Starting Y-coordinate for the gate in meters.
        MAX_GATE_WIDTH (float): Maximum width of the gate in meters.
        BALL_GATE_CENTER (float): Center position of the ball gate in meters.
        MIN_DIS_START (float): Minimum starting distance for objects in meters.
        MAX_DIS_START (float): Maximum starting distance for objects in meters.
        CAMERA_FOV (float): Field of view of the camera in radians.
        FOV_LINE_LENGTH (int): Length of the field of view lines in pixels.
        # Objects parameters
        TUBE_RADIUS (float): Radius of the tube in meters.
        TURTLE_RADIUS (float): Radius of the turtle in meters.
        BALL_RADIUS (float): Radius of the ball in meters.
    """
    

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
