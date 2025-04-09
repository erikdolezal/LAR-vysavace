import numpy as np
from planning.PathPlanningConfig import DataClasses
from planning.PathPlanningConfig import PlanningParm
from planning.PathPlanningConfig import ErrorCodes

class Planning:
    def __init__(self):
        self.objects = {}
        for type_ in DataClasses:
            self.objects[type_] = []
        self.goal_targer = None
        self.ball_pos = np.empty((0, 2))
        self.ball_path = None
        self.robot_pos = None

    def CreatPath(self, object_pos, test_alg = False):
        """Returns next point in a path or robot path and ball path if test_alg is true"""
        
        next_point = None
        # Objects identification 
        self.ClearObjects()
        inden_outcome = self.IdentifyObjects(object_pos)
        if inden_outcome == ErrorCodes.MORE_BLUE_ERR:
            return ErrorCodes.MORE_BLUE_ERR
        elif inden_outcome == ErrorCodes.NO_BALL_ERR:
            next_point = self.TurnRobotAround()

        elif inden_outcome == ErrorCodes.ZERO_BLUE_ERR:
            pass
            path_to_ball = self.GenerateTrajectory(self.robot_pos[0:2], self.ball_pos) #Generates a path to the ball
            if path_to_ball is None:
                return ErrorCodes.BALL_STUCK_ERR
            if test_alg:
                return path_to_ball, np.empty((0, 2)), np.empty((0, 2)) # Returns all paths for testing
            else:
                next_point = path_to_ball[1]
        else:
            self.ball_path = self.GenerateTrajectory(self.ball_pos, self.goal_targer) #Generates a path for the ball to the goal targer
            if self.ball_path is None:
                return ErrorCodes.BALL_STUCK_ERR
            
            shoot_path = self.GenerateShootPath() # Generates a shooting point for the robot and path to the ball
            
            if shoot_path is None:
                return ErrorCodes.NO_SHOOT_ERR
            
            # TODO: check if close to ball to shoot
            robot_destination = shoot_path[0]
            robot_path_to_shoot = self.GenerateTrajectory(self.objects[DataClasses.TURTLE][0][:2], robot_destination) #Generates a path for the robot to the shooting point
            if robot_path_to_shoot is None:
                return ErrorCodes.NO_RORBOT_ERR
            
            if test_alg:
                return robot_path_to_shoot, shoot_path, self.ball_path # Returns all paths for testing
            next_point = robot_path_to_shoot[1] # Sets the next point for the robot to follow
            
        return next_point # Returns the next point in the path for the robot to follow
    
    def ClearObjects(self):
        """Clears all objectss"""
        
        for type_ in DataClasses:
            self.objects[type_] = []
    
    def TurnRobotAround(self):
        """Turns the robot around by 180 degrees"""
        turn_point = None
        
        # TODO: implement the turn point
        
        return turn_point
    
    def IdentifyObjects(self, objects_in):
        """Indenfify objects and add them to the objects list"""
        
        center_sum = np.array([0, 0])
        blue_count = 0
        poss_balls = np.empty((0, 2))
        
        for object in objects_in:
            self.objects[object[2]].append(object)
            if object[2] == DataClasses.BLUE:
                blue_count += 1
                center_sum = center_sum + np.array(object[0:2])
            elif object[2] == DataClasses.BALL:
                poss_balls = np.vstack([poss_balls, np.array(object[0:2])]) 
            elif object[2] == DataClasses.TURTLE:
                #self.robot_pos = np.array([object[0:2], object[3]])
                pass
                
        self.ball_pos =  np.mean(poss_balls, axis=0) # Calculates the mean of all ball positions if the possition is not sure
        if self.ball_pos is None:
            return ErrorCodes.NO_BALL_ERR        
                
        # TODO: what to do if there are more than 2 blue tubes
        if blue_count > 2:
            return ErrorCodes.MORE_BLUE_ER
        elif blue_count == 2:
            self.goal_targer = center_sum/2 # Calculates the center of the goal
        elif blue_count == 1:
            self.goal_targer = center_sum # Sets a position of the blue tube as a goal target
        else:
            if self.goal_targer is None:
                return ErrorCodes.ZERO_BLUE_ERR
            else:
                return ErrorCodes.OK_ERR        
        return ErrorCodes.OK_ERR
    
    def GenerateTrajectory(self, start, target):
        """Gernerates a path from start to target point and avoids one obstacle point"""
        
        path_points = None
        problem_point = self.CheckColisions(np.array([start, target])) #Checks for colision of a direct path between start and target
        
        if problem_point is not None:
            path_points = self.GenerateWayAround(start, target, problem_point) #Generates a path around the problematic point
        else:
            path_points = np.array([start, target]) # Creates direct path to the target
            
        return path_points
    
    def GenerateWayAround(self, start, end, problem):
        """Generates a path around the problem point by finding the shortest path through the tangent points."""
        
        ## Finds the tangent points from the start and end points to the problem point
        tangent_point1, tangent_point2 = self.find_tangent_points(start, problem, PlanningParm.CLEARANCE)
        tangent_point3, tangent_point4 = self.find_tangent_points(end, problem, PlanningParm.CLEARANCE)
        
        # Creates a list of all possible combinations of tangent points``
        combinations = [(tangent_point1, tangent_point3), (tangent_point1, tangent_point4), (tangent_point2, tangent_point3), (tangent_point2, tangent_point4)]
        
        shortest_inter, min_distance = None, float('inf')
        for tangent_point_A, tangent_point_B in combinations:
            k1, q1 = self.line_equation(start, tangent_point_A) # start, A tangent line
            k2, q2 = self.line_equation(end, tangent_point_B) # end, B tangent line
            intersection = self.find_intersection(k1, q1, k2, q2) # finds the intersection point of the two tangent lines
            distance = self.path_length([start, tangent_point_A, intersection, tangent_point_B, end])
            if distance < min_distance:
                min_distance = distance
                shortest_inter = intersection
                            
        return np.array([start, shortest_inter, end])
    
    def GenerateShootPath(self):
        """Generates a shooting point for the robot and a path to the ball"""
        
        ball_point = self.ball_path[0]
        end_point = self.ball_path[1]
        direction = end_point - ball_point
        norm_direction = direction / np.linalg.norm(direction)
        shooting_point = ball_point - norm_direction * PlanningParm.SHOOT_STEPBACK # Calculates the shooting point by moving back from the ball point
        overshoot_point = ball_point + norm_direction * PlanningParm.SHOOT_SCALING # Calculates the overshoot point to gain speed
        
        return np.array([shooting_point, overshoot_point])
    
    def CheckColisions(self, path):
        """ Checks if there are any collisions with the path between the start and end points and returns the collision point if there is one."""
        
        colision_point = None
        
        for tube in self.objects[DataClasses.GREEN]: #Checks for colision with all green tubes
            tube_pos = np.array(tube[0:2])
            distance = self.point_distance_from_line(tube_pos, path[0], path[1]) #Calculates the distance from the line to the tube
            if distance < PlanningParm.CLEARANCE:
                colision_point = tube_pos
                break
            
        return colision_point
    
    def point_distance_from_line(self, point, A, B):
        """
        Calculates the perpendicular distance of a point from the line defined by points A and B.
        """
        
        AB = B - A
        AB_norm = np.linalg.norm(AB)
        distance = np.abs(np.cross(AB, point - A) / AB_norm)
        
        return distance
    
    def find_tangent_points(self, start, center, radius):
        """
        Finds two tangent points from start point to circle.
        """
        
        d = start - center
        d_2 = np.dot(d, d)
        r_2 = radius**2
        h_2 = d_2 - r_2
        a = np.sqrt(h_2)
        b = np.sqrt(h_2 * r_2) / d_2
        tangent_point1 = center + a * d + b * np.array([d[1], -d[0]])
        tangent_point2 = center + a * d - b * np.array([d[1], -d[0]])
        
        return tangent_point1, tangent_point2
    
    def line_equation(self, start, end):
        """Returns the slope and the absolute term of a line equation"""
        
        k = (start[1] - end[1]) / (end[0] - start[0])
        q = start[1] - k * end[0]
        
        return k, q
    
    def path_length(self, points):
        """Calculates the total length of a line."""
        
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    
    def find_intersection(self, k1, q1, k2, q2):
        """Finds the intersection point of two lines."""
        
        x = (q2 - q1) / (k1 - k2)
        y = k1 * x + q1
        return np.array([x, y])
    
    def are_points_in_proximity(self, point1, point2):
        """
        Checks if two points are within a given proximity.
        """
        state = False
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        if distance < PlanningParm.BALL_PROXIMITY:
            state = True
        return state
    
