import pygame
import numpy as np
from Sim.SimConfig import SimParm
from Sim.SimConfig import DataClasses


class Objects:
    """
    A class to represent objects in a simulation.
    Attributes:
        sim (object): The simulation instance the object belongs to.
        type (str, optional): The type of the object. Defaults to None.
        RADIUS (int): The radius of the object. Defaults to 0.
        pos (numpy.ndarray): The position of the object as a 2D array. Initialized if x and y are provided.
    Methods:
        DimToPixels(dim_real):
            Converts a real-world dimension to pixels based on simulation parameters.
        was_collision(collission_pos, radius):
            Checks if a collision occurred with another object based on its position and radius.
        get_position():
            Returns the current position of the object.
    """

    def __init__(self, sim, type=None, x=None, y=None):

        # object parametrs
        self.sim = sim
        self.type = type
        self.RADIUS = 0
        if x is not None and y is not None:
            self.pos = np.array([x, y])

    def DimToPixels(self, dim_real):
        """
        Converts a real-world dimension to its equivalent in pixels.
        Args:
            dim_real (float): The real-world dimension to be converted.
        Returns:
            int: The dimension in pixels, calculated based on the ratio of
                 simulation width in pixels to the real-world side length.
        """
        return int((SimParm.WIDTH / SimParm.SIDE_REAL) * dim_real)

    def was_collision(self, collission_pos, radius):
        """
        Determines whether a collision has occurred between the object and another object.
        Args:
            collission_pos (numpy.ndarray): The position of the other object as a NumPy array.
            radius (float): The radius of the other object.
        Returns:
            bool: True if a collision has occurred, False otherwise.
        """
        distance = np.linalg.norm(collission_pos - self.pos)
        if distance <= self.RADIUS + radius:
            return True
        else:
            return False

    def get_position(self):
        """
        Retrieve the current position of the object.
        Returns:
            tuple: The current position of the object as a tuple (e.g., (x, y)).
        """

        return self.pos


class PlayGround(Objects):
    """
    A class representing the playground in the simulation.
    Inherits from:
        Objects: The base class for simulation objects.
    Attributes:
        sim (object): The simulation instance to which the playground belongs.
        type (str, optional): The type of the object. Defaults to None.
        x (float, optional): The x-coordinate of the object. Defaults to None.
        y (float, optional): The y-coordinate of the object. Defaults to None.
    Methods:
        __init__(sim, type=None, x=None, y=None):
            Initializes a PlayGround object with the given simulation instance,
            type, and coordinates.
        draw():
            Draws the playground on the simulation screen. This includes filling
            the screen with a white background and drawing lines to represent
            specific areas of the playground.
    """

    def __init__(self, sim, type=None, x=None, y=None):
        super().__init__(sim, type, x, y)

    def draw(self):
        """
        Draws the simulation environment on the screen.
        This method clears the screen, draws dividing lines, and highlights specific
        regions of interest in the simulation using colored lines.
        - Fills the screen with a white background.
        - Draws a black vertical line in the middle of the screen.
        - Draws two horizontal red lines at specific positions determined by the
          `DimToPixels` method.
        The positions and colors of the lines are defined by the `SimParm` class.
        """

        self.sim.screen.fill(SimParm.WHITE)
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (SimParm.WIDTH / 2, 0), (SimParm.WIDTH / 2, SimParm.WIDTH), 3)
        pygame.draw.line(self.sim.screen, SimParm.RED, (0, self.DimToPixels(1.3)), (SimParm.WIDTH, self.DimToPixels(1.3)), 3)
        pygame.draw.line(self.sim.screen, SimParm.RED, (0, self.DimToPixels(2.1)), (SimParm.WIDTH, self.DimToPixels(2.1)), 3)


class Turtle(Objects):
    """
    Represents a TurtleBot object in the simulation.
    Attributes:
        angle (float): The current orientation of the TurtleBot in radians.
        last_velosity (float): The last recorded velocity of the TurtleBot.
        RADIUS (float): The radius of the TurtleBot, defined in simulation parameters.
    Methods:
        __init__(sim, type, x, y, angle):
            Initializes a TurtleBot object with its simulation context, type, position, and angle.
        draw():
            Draws the TurtleBot on the simulation screen, including its position, orientation, and field of view.
        move(velocity, angular_velocity, forward=True, clockwise=True):
            Moves the TurtleBot based on the given velocity and angular velocity.
            The direction of movement can be controlled using the `forward` and `clockwise` flags.
        get_info():
            Returns the current state of the TurtleBot as a NumPy array, including position, angle, and last velocity.
    """

    def __init__(self, sim, type, x, y, angle):
        super().__init__(sim, type, x, y)
        # turlte parametrs
        self.angle = angle
        self.last_velosity = 0
        # turtle constants
        self.RADIUS = SimParm.TURTLE_RADIUS

    def draw(self):
        """
        Draws the TurtleBot on the simulation screen.
        This method visualizes the TurtleBot's position, orientation, and field of view (FOV)
        on the simulation screen using Pygame. It includes:
        - A circle representing the TurtleBot's body.
        - A line indicating the front direction of the TurtleBot.
        - Two lines representing the edges of the TurtleBot's camera field of view.
        The drawing is scaled and positioned based on the TurtleBot's current position,
        orientation, and dimensions.
        Attributes:
            self.pos (tuple): The (x, y) position of the TurtleBot in the simulation.
            self.angle (float): The orientation angle of the TurtleBot in radians.
            self.RADIUS (float): The radius of the TurtleBot.
            self.sim.screen (pygame.Surface): The Pygame surface where the TurtleBot is drawn.
            SimParm.RED (tuple): The RGB color for the TurtleBot's body.
            SimParm.BLACK (tuple): The RGB color for the lines.
            SimParm.CAMERA_FOV (float): The field of view angle of the TurtleBot's camera in radians.
            SimParm.FOV_LINE_LENGTH (float): The length of the FOV lines.
        Note:
            The `DimToPixels` method is used to convert dimensions from simulation units
            to pixel units for rendering on the screen.
        """

        # Robot's front direction (for simulation of movement)
        line_length = self.DimToPixels(self.RADIUS) * 2
        line_end_x = self.DimToPixels(self.pos[0]) + np.cos(self.angle) * line_length
        line_end_y = self.DimToPixels(self.pos[1]) - np.sin(self.angle) * line_length

        fov_line_x1 = self.DimToPixels(self.pos[0]) + np.cos(self.angle + SimParm.CAMERA_FOV / 2) * SimParm.FOV_LINE_LENGTH
        fov_line_y1 = self.DimToPixels(self.pos[1]) - np.sin(self.angle + SimParm.CAMERA_FOV / 2) * SimParm.FOV_LINE_LENGTH
        fov_line_x2 = self.DimToPixels(self.pos[0]) + np.cos(self.angle - SimParm.CAMERA_FOV / 2) * SimParm.FOV_LINE_LENGTH
        fov_line_y2 = self.DimToPixels(self.pos[1]) - np.sin(self.angle - SimParm.CAMERA_FOV / 2) * SimParm.FOV_LINE_LENGTH

        pygame.draw.circle(self.sim.screen, SimParm.RED, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(self.RADIUS))
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), (line_end_x, line_end_y), 3)
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), (fov_line_x1, fov_line_y1), 4)
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), (fov_line_x2, fov_line_y2), 4)

    def move(self, velocity, angular_velocity, forward=True, clockwise=True):
        """
        Moves the robot based on the specified velocity and angular velocity.
        Parameters:
            velocity (float): The linear velocity of the robot.
            angular_velocity (float): The angular velocity of the robot.
            forward (bool, optional): If True, the robot moves forward. If False, the robot moves backward. Defaults to True.
            clockwise (bool, optional): If True, the robot rotates clockwise. If False, the robot rotates counterclockwise. Defaults to True.
        Updates:
            - The robot's angle is adjusted based on the angular velocity and direction.
            - The robot's position is updated based on the linear velocity, direction, and current angle.
            - The `last_velocity` attribute is set to the provided velocity.
        """

        if clockwise:
            self.angle += angular_velocity / SimParm.SIM_FPS
        else:
            self.angle -= angular_velocity / SimParm.SIM_FPS
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

        if forward:
            self.pos[0] += velocity / SimParm.SIM_FPS * np.cos(self.angle)
            self.pos[1] -= velocity / SimParm.SIM_FPS * np.sin(self.angle)
        else:
            self.pos[0] -= velocity / SimParm.SIM_FPS * np.cos(self.angle)
            self.pos[1] += velocity / SimParm.SIM_FPS * np.sin(self.angle)

        self.last_velosity = velocity

    def get_info(self):
        """
        Retrieve information about the object's current state.
        Returns:
            numpy.ndarray: A 1D array containing the following elements:
                - self.pos[0] (float): The x-coordinate of the object's position.
                - self.pos[1] (float): The y-coordinate of the object's position.
                - self.angle (float): The object's current angle or orientation.
                - self.last_velosity (float): The object's last recorded velocity.
        """

        return np.array([self.pos[0], self.pos[1], self.angle, self.last_velosity])


class Tube(Objects):
    """
    Represents a tube object in the simulation.
    Attributes:
        color (tuple or None): The color of the tube, represented as an RGB tuple. Defaults to None.
        RADIUS (float): The radius of the tube, defined as a constant from SimParm.TUBE_RADIUS.
    Methods:
        __init__(sim, type, x, y, color=None):
            Initializes a Tube object with its simulation context, type, position, and optional color.
        draw():
            Draws the tube on the simulation screen as a circle.
    """

    def __init__(self, sim, type, x, y, color=None):
        super().__init__(sim, type, x, y)

        # Tube parametrs
        self.color = color

        # Tube constants
        self.RADIUS = SimParm.TUBE_RADIUS

    def draw(self):
        """
        Draws the object as a circle on the simulation screen.
        This method uses the pygame library to draw a circle representing the object.
        The circle's position, color, and radius are determined by the object's attributes.
        The position and radius are converted from simulation dimensions to pixel dimensions
        using the `DimToPixels` method.
        Attributes:
            self.sim.screen (pygame.Surface): The surface on which the object is drawn.
            self.color (tuple): The color of the circle in RGB format.
            self.pos (tuple): The position of the object in simulation dimensions.
            self.RADIUS (float): The radius of the object in simulation dimensions.
        """

        pygame.draw.circle(self.sim.screen, self.color, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(self.RADIUS))


class Ball(Objects):
    def __init__(self, sim, type, x, y, color=None):
        super().__init__(sim, type, x, y)
        self.RADIUS = SimParm.BALL_RADIUS
        if color is None:
            self.color = SimParm.YELLOW
        self.velocity = np.array([0, 0])

    def draw(self):
        """ Draw the objects. """
        pygame.draw.circle(self.sim.screen, self.color, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(self.RADIUS))

    def movement(self):
        self.pos[0] += self.velocity[0] / SimParm.SIM_FPS
        self.velocity[0] = self.velocity[0] * SimParm.SIM_FRICTION
        self.pos[1] += self.velocity[1] / SimParm.SIM_FPS
        self.velocity[1] = self.velocity[1] * SimParm.SIM_FRICTION

    def set_velocity(self, new_velocity):
        self.velocity = new_velocity


class Point(Objects):
    class Point:
        """
        A class representing a point object in a simulation.
        Attributes:
            sim (object): The simulation instance the point belongs to.
            type (str): The type of the point object.
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            color (tuple): The color of the point, default is SimParm.BLACK.
        Methods:
            __del__():
                Destructor for the Point class, cleans up resources.
            set_position(x, y):
                Sets the position of the point using the given x and y coordinates.
            draw():
                Draws the point on the simulation screen as a circle.
        """

    def __init__(self, sim, type, x, y, color=SimParm.BLACK):
        super().__init__(sim, type, x, y)
        self.color = color
        self.x = x
        self.y = y

    def __del__(self):
        """ Destructor for Point class. """
        del self.pos

    def set_position(self, x, y):
        """
        Set the position of the object.
        Parameters:
        x (float): The x-coordinate of the new position.
        y (float): The y-coordinate of the new position.
        Returns:
        None
        """

        self.pos = np.array([x, y])

    def draw(self):
        """
        Draws the object on the simulation screen as a circle.
        This method uses the pygame library to render a circle representing the object.
        The circle's position, color, and size are determined by the object's attributes.
        The position is converted from simulation dimensions to pixel coordinates using
        the `DimToPixels` method. The radius of the circle is fixed at 0.1 in simulation
        dimensions, which is also converted to pixels.
        Returns:
            None
        """

        pygame.draw.circle(self.sim.screen, self.color, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(0.1))


class Path(Objects):
    class Path:
        """
        A class representing a path consisting of multiple points.
        Attributes:
            points (list): A list of Point objects representing the path.
            color (tuple): The color of the path, default is SimParm.BLACK.
            sim (object): The simulation object associated with the path.
        Methods:
            __del__():
                Destructor for the Path class. Cleans up the points list.
            add_point(point):
                Adds a new point to the path.
                Args:
                    point (tuple): A tuple containing the x and y coordinates of the point.
            draw():
                Draws the path by connecting the points with lines on the simulation screen.
        """

    def __init__(self, sim, color=SimParm.BLACK):
        self.points = []
        self.color = color
        self.sim = sim

    def __del__(self):
        """ Destructor for Path class. """
        del self.points

    def add_point(self, point):
        """
        Adds a new point to the list of points.
        Args:
            point (tuple): A tuple containing the x and y coordinates of the point
                           to be added (e.g., (x, y)).
        Creates:
            A new `Point` object with the specified coordinates, associated with
            the current simulation (`sim`), and appends it to the `points` list.
        """

        point_new = Point(sim=self.sim, type=DataClasses.POINT, x=point[0], y=point[1], color=self.color)
        self.points.append(point_new)

    def draw(self):
        """
        Draws lines connecting a sequence of points on the simulation screen.
        This method iterates through the list of points and draws a line between
        each consecutive pair of points. The lines are drawn using the specified
        color and a fixed thickness.
        Uses the `DimToPixels` method to convert the coordinates of the points
        from simulation dimensions to pixel dimensions.
        Args:
            None
        Returns:
            None
        """

        for index in range(len(self.points) - 1):
            pygame.draw.line(self.sim.screen, self.color, (self.DimToPixels(self.points[index].x), self.DimToPixels(self.points[index].y)), (self.DimToPixels(self.points[index + 1].x), self.DimToPixels(self.points[index + 1].y)), 3)
