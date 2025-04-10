import pygame
import numpy as np
import random
from Sim.SimObjects import *
from Sim.SimConfig import SimParm
from Sim.SimConfig import DataClasses
from planning.PathPlanning import Planning

class Simulation:
    
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
        return random.choice([random.uniform(a, b), random.uniform(-b, -a)])
    
    def check_collisions(self, pos, radius, skip=[]):
        for type_ in self.objects:
            if type_ in skip: continue
            for inst in self.objects[type_]:
                if inst.was_collision(pos, radius):
                    return True
        return False
    
    def generate_gate(self):
        """Generate a two blue pillars, which are gate"""
        first_y = SimParm.GATE_Y_START
        first_x = random.uniform(1, SimParm.SIDE_REAL/2) #Border is set to 1 meter
        second_y = SimParm.GATE_Y_START
        second_x = first_x + self.generate_point_in_ring(SimParm.MAX_GATE_WIDTH, SimParm.SIDE_REAL/2-1)
        first_tube = Tube(self, DataClasses.BLUE, first_x, first_y, SimParm.BLUE)
        second_tube = Tube(self, DataClasses.BLUE, second_x, second_y, SimParm.BLUE)
        self.objects[DataClasses.BLUE] = [first_tube, second_tube]
        
        
    def generate_ball(self):
        """Generate a ball according to the rules"""
        gate_center = (self.objects[DataClasses.BLUE][0].pos[0] + self.objects[DataClasses.BLUE][1].pos[0])/2
        x = gate_center + self.generate_point_in_ring(0, SimParm.BALL_GATE_CENTER)
        y =self.objects[DataClasses.BLUE][0].pos[1] + random.uniform(SimParm.MIN_DIS_START, SimParm.MIN_DIS_START*4)
        self.objects[DataClasses.BALL] = [Ball(self, DataClasses.BALL, x, y)]       
        
    def generate_tube(self, count):
        pos = np.empty(2)
        iter = 0
        while iter < count:
            pos = np.array([random.uniform(0, SimParm.SIDE_REAL), random.uniform(SimParm.GATE_Y_START, SimParm.SIDE_REAL)])
            if self.check_collisions(pos, 0.5) == False:
                iter+=1
                self.objects[DataClasses.GREEN].append(Tube(self, DataClasses.GREEN, pos[0], pos[1], SimParm.GREEN))
                
                
    def generate_turtle(self):
        """Generate turtle according to the rules"""
        pos = np.empty(2)
        while True:
            pos = np.array([random.uniform(0, SimParm.SIDE_REAL), random.uniform(SimParm.MAX_DIS_START, SimParm.SIDE_REAL)])
            if self.check_collisions(pos, SimParm.MIN_DIS_START) == False:
                break
        self.objects[DataClasses.TURTLE]=[(Turtle(self, DataClasses.TURTLE, pos[0], pos[1], np.pi/2))]
        
    def draw_object(sefl, object):
        for sub in object:
            sub.draw()
            
    def redraw_everything(self):
        self.play_ground.draw()
        for type_ in self.objects:
            self.draw_object(self.objects[type_])
        self.redraw_paths()
        
        
    def redraw_paths(self):
        if self.path_robot is not None:
            self.path_robot.draw()
            #del self.path_robot
            #self.path_robot = None
        if self.path_ball is not None:
            self.path_ball.draw()
            #del self.path_ball
            #self.path_ball = None
        if self.path_shoot is not None:
            self.path_shoot.draw()
            #del self.path_shoot
            #self.path_shoot = None
            
        
    def maual_movement(self):
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
        output = np.empty((0, 3))
        for class_type in self.objects:
            if class_type == DataClasses.TURTLE:
                pass
            else:
                for object_iter in self.objects[class_type]:
                    output = np.vstack([output, np.array([object_iter.pos[0] - SimParm.SIM_OFFSET, object_iter.pos[1] - SimParm.SIM_OFFSET, object_iter.type])])
        return output
    
    def planing_visual(self):
        robot_pos = self.objects[DataClasses.TURTLE][0].get_info()[:3] - np.array([SimParm.SIM_OFFSET, SimParm.SIM_OFFSET, 0])
        robot_path, shoot_path, ball_path  = self.path.creat_path(self.get_positions(), robot_pos, True)
        if robot_path is not None:
            self.path_robot = self.load_path(robot_path, SimParm.RED)
        else: return False
        if shoot_path is not None:
            self.path_shoot = self.load_path(shoot_path, SimParm.BLUE)
        else: return False
        if ball_path is not None:
            self.path_ball = self.load_path(ball_path, SimParm.YELLOW)
        else: return False
        return True
    
    def load_path(self, path, color):
        """Loads the path from a given path"""
        
        exit_path = Path(self, color=color)
        for point in path:
            exit_path.add_point(point + np.array([SimParm.SIM_OFFSET, SimParm.SIM_OFFSET]))
            
        return exit_path
          
    def main(self):
        running = True
        self.generate_gate()
        self.generate_ball()
        self.generate_tube(2)
        self.generate_turtle()
        #Moving logic
            # TODO
            
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                running = False
            
            self.sim_update()
            self.planing_visual()
            self.redraw_everything()
            self.maual_movement()
            self.clock.tick(100)
            # Update display
            pygame.display.flip()
            
        pygame.quit()

if __name__ == "__main__":
    Sim = Simulation()
    Sim.main()
