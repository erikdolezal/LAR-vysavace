import pygame
import numpy as np
import random
from Sim.SimObjects import *
from Sim.SimConfig import SimParm
from Sim.SimConfig import DataClasses
from planning.PathPlanning import Planning

class Simulation:
    """
    Simulation class for a 2D TurtleBot football simulator.
    This class handles the initialization, object generation, collision detection, 
    movement, and visualization of the simulation. It uses pygame for rendering 
    and numpy for mathematical operations.
    Methods:
        __init__():
            Initializes the simulation, pygame, and sets up the environment.
        generate_point_in_ring(a, b):
            Generates a random point within a ring defined by radii `a` and `b`.
        check_collisions(pos, radius, skip=[]):
            Checks for collisions of a circular object with other objects in the simulation.
        generate_gate():
            Generates a pair of blue pillars representing the gate.
        generate_ball():
            Generates a ball according to the simulation rules.
        generate_tube(count):
            Generates a specified number of green tubes at random positions.
        generate_turtle():
            Generates a turtle robot at a random position.
        draw_object(object):
            Draws a given object on the screen.
        redraw_everything():
            Redraws the playground, all objects, and paths.
        redraw_paths():
            Redraws the paths for the robot, ball, and shooting trajectory.
        maual_movement():
            Handles manual movement of the turtle robot using keyboard inputs.
        sim_update():
            Updates the simulation, including ball movement and collision handling.
        get_positions():
            Retrieves the positions and types of all objects in the simulation.
        is_in_fov(objects, robot_pos, robot_angle):
            Checks if objects are within the field of view of the robot.
        planing_visual():
            Runs the planning algorithm and visualizes the generated paths.
        load_path(path, color):
            Loads a path and visualizes it with the specified color.
        main():
            Main loop of the simulation. Handles events, updates, and rendering.
    """
    
    def __init__(self):        
        # Initialize pygame
        pygame.init()

        # Screen size and setup
        self.screen = pygame.display.set_mode((SimParm.WIDTH, SimParm.HEIGHT))
        pygame.display.set_caption("TurtleBot 2D Simulator for Football")
        
        # Real playground setup
        self.objects = {}
        self.objects[DataClasses.GREEN] = []
        self.play_ground = PlayGround(self)
        self.path = Planning()
        self.path_robot = None
        self.path_ball = None
        self.path_shoot = None
            
        # Clock for framerate
        self.clock = pygame.time.Clock()
            
    def generate_point_in_ring(self, a, b):
        """
        Generates a random point within a ring defined by two radii, `a` and `b`.

        The ring is defined as the area between two concentric circles with radii `a` and `b`.
        The function randomly selects a value within the range [a, b] or [-b, -a].

        Args:
            a (float): The inner radius of the ring.
            b (float): The outer radius of the ring.

        Returns:
            float: A randomly generated point within the specified ring.
        """

        return random.choice([random.uniform(a, b), random.uniform(-b, -a)])
    
    def check_collisions(self, pos, radius, skip=[]):
        """
        Checks for collisions between a given position and radius against all objects in the simulation.
        Args:
            pos (tuple): The position to check for collisions, typically a (x, y) coordinate.
            radius (float): The radius around the position to check for collisions.
            skip (list, optional): A list of object types to skip during the collision check. Defaults to an empty list.
        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        
        for type_ in self.objects:
            if type_ in skip: continue
            for inst in self.objects[type_]:
                if inst.was_collision(pos, radius):
                    return True
        return False
    
    def generate_gate(self):
        """
        Generates a gate consisting of two tubes placed horizontally on the simulation field.
        The method creates two tubes of the same color (blue) and places them at specific 
        coordinates along the Y-axis defined by `SimParm.GATE_Y_START`. The X-coordinates 
        of the tubes are determined randomly within a specified range, ensuring the gate 
        width does not exceed the maximum allowed value.
        The generated tubes are stored in the `self.objects` dictionary under the key 
        corresponding to their color.
        Returns:
            None
        """
        
        
        first_y = SimParm.GATE_Y_START
        first_x = random.uniform(1, SimParm.SIDE_REAL/2) #Border is set to 1 meter
        second_y = SimParm.GATE_Y_START
        second_x = first_x + self.generate_point_in_ring(SimParm.MAX_GATE_WIDTH, SimParm.SIDE_REAL/2-1)
        first_tube = Tube(self, DataClasses.BLUE, first_x, first_y, SimParm.BLUE)
        second_tube = Tube(self, DataClasses.BLUE, second_x, second_y, SimParm.BLUE)
        self.objects[DataClasses.BLUE] = [first_tube, second_tube]
        
        
    def generate_ball(self):
        """
        Generates a new ball object and places it in the simulation.
        The ball's position is determined based on the center of the blue gate and 
        a random offset within a specified range. The x-coordinate is calculated 
        by adding a random offset to the center of the blue gate, while the 
        y-coordinate is determined by adding a random value within a specified 
        range to the y-coordinate of the first blue object.
        The generated ball is then added to the simulation's objects dictionary 
        under the BALL key.
        Attributes:
            gate_center (float): The x-coordinate of the center of the blue gate.
            x (float): The x-coordinate of the generated ball.
            y (float): The y-coordinate of the generated ball.
        Raises:
            KeyError: If the required keys (BLUE or BALL) are not present in the 
                      objects dictionary.
        """
        
        
        gate_center = (self.objects[DataClasses.BLUE][0].pos[0] + self.objects[DataClasses.BLUE][1].pos[0])/2
        x = gate_center + self.generate_point_in_ring(0, SimParm.BALL_GATE_CENTER)
        y =self.objects[DataClasses.BLUE][0].pos[1] + random.uniform(SimParm.MIN_DIS_START, SimParm.MIN_DIS_START*4)
        self.objects[DataClasses.BALL] = [Ball(self, DataClasses.BALL, x, y)]       
        
    def generate_tube(self, count):
        """
        Generates a specified number of tube objects and places them in the simulation environment.
        Args:
            count (int): The number of tube objects to generate.
        The method randomly generates positions for the tubes within the simulation boundaries,
        ensuring that they do not collide with existing objects. If a valid position is found,
        the tube is added to the list of objects in the simulation.
        Notes:
            - The position is generated within the range defined by `SimParm.SIDE_REAL` for both
              x and y coordinates, with additional constraints for the y-coordinate to avoid
              overlap with the gate area.
            - Collision checks are performed using the `check_collisions` method with a radius
              of 0.5.
            - Tubes are appended to the `self.objects[DataClasses.GREEN]` list.
        """
        
        
        pos = np.empty(2)
        iter = 0
        while iter < count:
            pos = np.array([random.uniform(0, SimParm.SIDE_REAL), random.uniform(SimParm.GATE_Y_START+SimParm.MAX_DIS_START, SimParm.SIDE_REAL)])
            if self.check_collisions(pos, 0.5) == False:
                iter+=1
                self.objects[DataClasses.GREEN].append(Tube(self, DataClasses.GREEN, pos[0], pos[1], SimParm.GREEN))
                
                
    def generate_turtle(self):
        """
        Generates a new turtle object and places it in the simulation environment.
        The method randomly generates a position for the turtle within the simulation
        boundaries, ensuring that it does not collide with existing objects and maintains
        a minimum distance from other objects. Once a valid position is found, the turtle
        is created and added to the simulation's objects dictionary.
        Returns:
            None
        """
        
        pos = np.empty(2)
        while True:
            pos = np.array([random.uniform(0, SimParm.SIDE_REAL), random.uniform(SimParm.MAX_DIS_START, SimParm.SIDE_REAL)])
            if self.check_collisions(pos, SimParm.MIN_DIS_START) == False:
                break
        self.objects[DataClasses.TURTLE]=[(Turtle(self, DataClasses.TURTLE, pos[0], pos[1], np.pi/2))]
        
    def draw_object(sefl, object):
        """
        Draws the given object by iterating through its subcomponents and calling their draw method.
        Args:
            object (iterable): An iterable containing subcomponents, each of which must have a `draw` method.
        """
        
        for sub in object:
            sub.draw()
            
    def redraw_everything(self):
        """
        Redraws all elements in the simulation environment.
        This method updates the visual representation of the playground,
        all objects, and paths in the simulation. It performs the following steps:
        1. Draws the playground.
        2. Iterates through all object types and draws each object.
        3. Redraws the paths in the simulation.
        Note:
            Ensure that the `self.play_ground`, `self.objects`, and `self.redraw_paths`
            are properly initialized before calling this method.
        """
        
        
        self.play_ground.draw()
        for type_ in self.objects:
            self.draw_object(self.objects[type_])
        self.redraw_paths()
        
        
    def redraw_paths(self):
        """
        Redraws the paths for the robot, ball, and shoot if they exist.
        This method checks if the paths for the robot, ball, and shoot are not None.
        If a path exists, it calls the `draw` method on the respective path object
        to redraw it on the screen or canvas.
        """
        
        if self.path_robot is not None:
            self.path_robot.draw()
        if self.path_ball is not None:
            self.path_ball.draw()
        if self.path_shoot is not None:
            self.path_shoot.draw()
            
        
    def maual_movement(self):
        """
        Handles manual movement of the turtle object based on keyboard input.
        This method listens for specific key presses to control the movement
        and rotation of the turtle object in the simulation.
        Key Bindings:
            - W: Move the turtle forward.
            - S: Move the turtle backward.
            - A: Rotate the turtle counter-clockwise (turn left).
            - D: Rotate the turtle clockwise (turn right).
        Movement Details:
            - Forward and backward movement is controlled by the `move` method
              of the turtle object, with the `forward` parameter determining
              the direction.
            - Rotation is controlled by the `move` method, with the `clockwise`
              parameter determining the direction of rotation.
        Note:
            This method requires the `pygame` library for detecting key presses
            and assumes that the `self.objects` dictionary contains a `TURTLE`
            key mapped to a list of turtle objects.
        """
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:  # Move forward
            self.objects[DataClasses.TURTLE][0].move(1, 0)
        if keys[pygame.K_s]:  # Move backward
            self.objects[DataClasses.TURTLE][0].move(1, 0, forward=False)
        if keys[pygame.K_a]:  # Turn left (counter-clockwise)
            self.objects[DataClasses.TURTLE][0].move(0, 1)
        if keys[pygame.K_d]:  # Turn right (clockwise)
            self.objects[DataClasses.TURTLE][0].move(0, 1, clockwise=False)
            
    def sim_update(self):
        """
        Updates the simulation state by handling the movement of the ball and its interactions
        with other objects in the simulation.
        The method performs the following tasks:
        1. Updates the ball's movement by invoking its `movement` method.
        2. Checks for a collision between the ball and the turtle. If a collision occurs:
           - Adjusts the ball's velocity based on the turtle's orientation and velocity.
        3. Checks for collisions between the ball and other objects (excluding the ball and turtle).
           - If a collision is detected, the ball's velocity is set to zero.
        Note:
        - The ball's velocity is updated using the `set_velocity` method.
        - Collisions are detected using the `was_collision` and `check_collisions` methods.
        """        
        
        #update ball's movement
        self.objects[DataClasses.BALL][0].movement()
        current = self.objects[DataClasses.TURTLE][0].get_info() 
        if self.objects[DataClasses.BALL][0].was_collision(current[:2], self.objects[DataClasses.TURTLE][0].RADIUS):
            if np.pi <= current[2] < 3*np.pi/4:
                new_velocity = np.array([-current[3]*np.cos(current[2]),-current[3]*np.sin(current[2])])
                self.objects[DataClasses.BALL][0].set_velocity(new_velocity) 
            else:
                new_velocity = np.array([current[3]*np.cos(current[2]),-current[3]*np.sin(current[2])])
                self.objects[DataClasses.BALL][0].set_velocity(new_velocity) 
        if self.check_collisions(self.objects[DataClasses.BALL][0].pos, self.objects[DataClasses.BALL][0].RADIUS, [DataClasses.BALL, DataClasses.TURTLE]):
            self.objects[DataClasses.BALL][0].set_velocity(np.array([0,0]))
            
    def get_positions(self):
        """
        Retrieves the positions and types of objects in the simulation, excluding objects of type TURTLE.
        Returns:
            numpy.ndarray: A 2D array where each row represents an object. Each row contains:
                - The x-coordinate of the object's position adjusted by the simulation offset.
                - The y-coordinate of the object's position adjusted by the simulation offset.
                - The type of the object.
        """
        
        output = np.empty((0, 3))
        for class_type in self.objects:
            if class_type == DataClasses.TURTLE:
                pass
            else:
                for object_iter in self.objects[class_type]:
                    output = np.vstack([output, np.array([object_iter.pos[0] - SimParm.SIM_OFFSET, object_iter.pos[1] - SimParm.SIM_OFFSET, object_iter.type])])
        return output
    
    def is_in_fov(self, objects, robot_pos, robot_angle):
        """
        Determines which objects are within the field of view (FOV) of a robot.
        Args:
            objects (numpy.ndarray): A 2D array of shape (N, M) where each row represents an object.
                                    The first two columns (objects[:, :2]) represent the (x, y) positions of the objects.
            robot_pos (numpy.ndarray): A 1D array of shape (2,) representing the (x, y) position of the robot.
            robot_angle (float): The orientation of the robot in radians.
        Returns:
            numpy.ndarray: A subset of the input `objects` array containing only the objects that are within the robot's FOV.
        """
 
        positions = objects[:, :2]      # (N, 2)

        vecs_to_objects = positions - robot_pos
        angles_to_objects = -np.arctan2(vecs_to_objects[:, 1], vecs_to_objects[:, 0])
        angle_diffs = np.abs(angles_to_objects - robot_angle)
        within_fov = angle_diffs <= SimParm.CAMERA_FOV / 2

        visible = within_fov
        return objects[visible]
    
    def planing_visual(self):
        """
        Visualizes the planning paths for the robot, including its movement path, 
        shooting path, and ball path, based on the robot's position and objects 
        within its field of view (FOV).
        The method performs the following steps:
        1. Calculates the robot's position relative to a simulation offset.
        2. Identifies objects within the robot's field of view.
        3. Creates paths for the robot, shooting, and ball based on the identified objects.
        4. Loads and visualizes the paths if they are successfully created.
        Returns:
            bool: 
                - True if the robot path is not created (indicating no further visualization).
                - False if the robot path is successfully created and visualized.
        """
        
        
        robot_pos = self.objects[DataClasses.TURTLE][0].get_info()[:3] - np.array([SimParm.SIM_OFFSET, SimParm.SIM_OFFSET, 0])
        objects_in_fov = self.is_in_fov(self.get_positions(), robot_pos[:2], robot_pos[2])
        robot_path, shoot_path, ball_path  = self.path.create_path(objects_in_fov, robot_pos, True)
        
        if robot_path is not None:
            self.path_robot = self.load_path(robot_path, SimParm.RED)
        else: 
            return True
        if shoot_path is not None:
            self.path_shoot = self.load_path(shoot_path, SimParm.BLUE)
        if ball_path is not None:
            self.path_ball = self.load_path(ball_path, SimParm.YELLOW)
            
        return False
    
    def load_path(self, path, color):
        """
        Loads a path and creates an ExitPath object with the specified color.
        Args:
            path (list of numpy.ndarray): A list of points representing the path. 
                Each point is expected to be a numpy array.
            color (str): The color associated with the ExitPath object.
        Returns:
            ExitPath: An ExitPath object with the given points and color.
        """
        
        exit_path = Path(self, color=color)
        for point in path:
            exit_path.add_point(point + np.array([SimParm.SIM_OFFSET, SimParm.SIM_OFFSET]))
            
        return exit_path
          
    def main(self):
        """
        The main function that runs the simulation loop.
        This function initializes the simulation by generating various objects such as gates, balls, tubes, 
        and turtles. It then enters a loop where it processes events, updates the simulation, checks for 
        completion conditions, and redraws the simulation state. The loop continues running until the user 
        quits the application or the mission is accomplished.
        Key functionalities:
        - Initializes simulation objects.
        - Handles user input and events.
        - Updates the simulation state.
        - Checks for mission completion.
        - Redraws the simulation visuals.
        - Limits the frame rate to 100 frames per second.
        The simulation ends when the user closes the application, presses the ESCAPE key, or the mission 
        is successfully completed.
        Note:
        - This function requires the `pygame` library to be properly initialized before execution.
        - The `sim_update`, `planing_visual`, `redraw_everything`, and `maual_movement` methods are assumed 
          to be defined elsewhere in the class.
        """
        
        running = True
        
        # Set up the simulation - generate objects
        self.generate_gate()
        self.generate_ball()
        self.generate_tube(2)
        self.generate_turtle()
            
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                running = False
            
            self.sim_update()
            if self.planing_visual():
                print("Mission accomplished - ended with a goal!")
                running = False
            self.redraw_everything()
            self.maual_movement()
            self.clock.tick(100)
            # Update display
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    Sim = Simulation()
    Sim.main()
