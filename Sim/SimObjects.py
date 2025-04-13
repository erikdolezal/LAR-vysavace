import pygame
import numpy as np
import random
from Sim.SimConfig import SimParm
from Sim.SimConfig import DataClasses


class Objects:
    def __init__(self, sim, type=None, x=None, y=None):
        
        #object parametrs
        self.sim = sim
        self.type = type
        self.RADIUS = 0
        if x is not None and y is not None:
            self.pos = np.array([x, y])
    
    def DimToPixels(self, dim_real):
        """Recalculate real dimension to pixels. """
        return int((SimParm.WIDTH/SimParm.SIDE_REAL)*dim_real)  
    
    def was_collision(self, collission_pos, radius):
        distance = np.linalg.norm(collission_pos - self.pos)
        if distance <= self.RADIUS + radius:
            return True
        else:
            return False
    
    def get_position(self):
        return self.pos
    
class PlayGround(Objects):
    def __init__(self, sim, type=None, x=None, y=None):
        super().__init__(sim, type, x, y)
    
    def draw(self):
        self.sim.screen.fill(SimParm.WHITE)
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (SimParm.WIDTH/2, 0), (SimParm.WIDTH/2, SimParm.WIDTH), 3)
        pygame.draw.line(self.sim.screen, SimParm.RED, (0, self.DimToPixels(1.3)), (SimParm.WIDTH, self.DimToPixels(1.3)), 3)
        pygame.draw.line(self.sim.screen, SimParm.RED, (0, self.DimToPixels(2.1)), (SimParm.WIDTH, self.DimToPixels(2.1)), 3)
    
    
class Turtle(Objects):
    def __init__(self, sim, type, x, y, angle):
        super().__init__(sim, type, x, y)
        #turlte parametrs
        self.angle = angle
        self.last_velosity = 0
        #turtle constants
        self.RADIUS = SimParm.TURTLE_RADIUS
        
    def draw(self):
        """ Draw the TurtleBot at position (x, y) with angle. """
        # Robot's front direction (for simulation of movement)
        line_length = self.DimToPixels(self.RADIUS) * 2
        line_end_x = self.DimToPixels(self.pos[0]) + np.cos(self.angle) * line_length
        line_end_y = self.DimToPixels(self.pos[1]) - np.sin(self.angle) * line_length
        
        fov_line_x1 = self.DimToPixels(self.pos[0]) + np.cos(self.angle + SimParm.CAMERA_FOV/2) * SimParm.FOV_LINE_LENGTH
        fov_line_y1 = self.DimToPixels(self.pos[1]) - np.sin(self.angle + SimParm.CAMERA_FOV/2) * SimParm.FOV_LINE_LENGTH
        fov_line_x2 = self.DimToPixels(self.pos[0]) + np.cos(self.angle - SimParm.CAMERA_FOV/2) * SimParm.FOV_LINE_LENGTH
        fov_line_y2 = self.DimToPixels(self.pos[1]) - np.sin(self.angle - SimParm.CAMERA_FOV/2) * SimParm.FOV_LINE_LENGTH

        pygame.draw.circle(self.sim.screen, SimParm.RED, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(self.RADIUS))
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), (line_end_x, line_end_y), 3)
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), (fov_line_x1, fov_line_y1), 4)
        pygame.draw.line(self.sim.screen, SimParm.BLACK, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), (fov_line_x2, fov_line_y2), 4)
            
    def move(self, velocity, angular_velocity, forward=True, clockwise=True):
        """ Move the robot accorting to given velocity and angular velocity"""
        if clockwise:
            self.angle += angular_velocity/SimParm.SIM_FPS
        else:
            self.angle -= angular_velocity/SimParm.SIM_FPS
        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi
        
        if forward:
            self.pos[0] += velocity/SimParm.SIM_FPS * np.cos(self.angle)
            self.pos[1] -= velocity/SimParm.SIM_FPS * np.sin(self.angle) 
        else:
            self.pos[0] -= velocity/SimParm.SIM_FPS * np.cos(self.angle)
            self.pos[1] += velocity/SimParm.SIM_FPS * np.sin(self.angle)
            
        self.last_velosity = velocity
    
    def get_info(self):
        return np.array([self.pos[0], self.pos[1], self.angle, self.last_velosity])
    
class Tube(Objects):
    def __init__(self, sim, type, x, y, color=None):
        super().__init__(sim, type, x, y)
        
        #Tube parametrs
        self.color = color
        
        #Tube constants
        self.RADIUS = SimParm.TUBE_RADIUS
        
    def draw(self):
        """ Draw the objects. """
        pygame.draw.circle(self.sim.screen, self.color, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(self.RADIUS))
        
        
class Ball(Objects):
    def __init__(self, sim, type, x, y, color=None):
        super().__init__(sim, type, x, y)
        self.RADIUS = SimParm.BALL_RADIUS
        if color is None:
            self.color = SimParm.YELLOW
        self.velocity = np.array([0,0])           
    def draw(self):
        """ Draw the objects. """
        pygame.draw.circle(self.sim.screen, self.color, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(self.RADIUS))
        
    def movement(self):
        self.pos[0] += self.velocity[0]/SimParm.SIM_FPS
        self.velocity[0] = self.velocity[0]*SimParm.SIM_FRICTION
        self.pos[1] += self.velocity[1]/SimParm.SIM_FPS
        self.velocity[1] = self.velocity[1]*SimParm.SIM_FRICTION
        
    def set_velocity(self, new_velocity):
        self.velocity = new_velocity
    
class Point(Objects):
    def __init__(self, sim, type, x, y, color=SimParm.BLACK):
        super().__init__(sim, type, x, y)
        self.color = color
        self.x = x
        self.y = y        
    def __del__(self):
        """ Destructor for Point class. """
        
        del self.pos
        
    def set_position(self, x, y):
        self.pos = np.array([x, y])
        
    def draw(self):
        """ Draw the objects. """
        pygame.draw.circle(self.sim.screen, self.color, (self.DimToPixels(self.pos[0]), self.DimToPixels(self.pos[1])), self.DimToPixels(0.1))
        
class Path(Objects):
    def __init__(self, sim, color=SimParm.BLACK):
        self.points = []
        self.color = color
        self.sim = sim
    
    def __del__(self):
        """ Destructor for Path class. """
        del self.points
    
    def add_point(self, point):
        """ Add a point to the path. """
        point_new = Point(sim=self.sim,type=DataClasses.POINT ,x=point[0], y=point[1], color=self.color)
        self.points.append(point_new)
        
    def draw(self):
        """ Draw the objects. """
        for index in range(len(self.points)-1):
            pygame.draw.line(self.sim.screen, self.color, (self.DimToPixels(self.points[index].x), self.DimToPixels(self.points[index].y)), (self.DimToPixels(self.points[index+1].x), self.DimToPixels(self.points[index+1].y)), 3)
