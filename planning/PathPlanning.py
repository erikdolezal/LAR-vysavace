import numpy as np
from PathPlanningConfig import *

class Planning:
    def __init__(self):
        self.objects = {}
        for type_ in DataClasses:
            self.objects[type_] = []
        self.goal_center = np.array([0, 0])
        self.ball_path = None
        self.robot_path = None

    def CreatPath(self, object_pos, test_alg = False):
        self.ClearObjects()
        if not self.IdentifyObjects(object_pos):
            return None
        self.ball_path = self.GenerateBallPath()
        if self.ball_path is None:
            return None
        shoot_path = self.GenerateShootPath()
        if shoot_path is None:
            return None
        robot_destination = shoot_path[1]
        if self.are_points_in_proximity(robot_destination, self.objects[DataClasses.TURTLE][0][0:2]): 
            self.robot_path = np.array([[0,0], shoot_path[1]])
        else:
            robot_obstiacle = self.CheckColisions(np.array([self.objects[DataClasses.TURTLE][0][0:2], robot_destination]))
            if robot_obstiacle is not None:
                self.robot_path = self.GenerateWayAround(self.objects[DataClasses.TURTLE][0][0:2], robot_destination, robot_obstiacle)
            else:
                self.robot_path = np.array([self.objects[DataClasses.TURTLE][0][0:2]])
                self.robot_path = np.vstack([self.robot_path, shoot_path])
        final_out = None
        if test_alg is True:
            final_out = self.robot_path
        else:
            final_out = self.robot_path[1]
        return final_out
    
    def ClearObjects(self):
        for type_ in DataClasses:
            self.objects[type_] = []
        self.goal_center = np.array([0, 0])
        self.ball_path = None
        self.robot_path = None
    
    def IdentifyObjects(self, objects_in):
        center_sum = np.array([0, 0])
        blue_count = 0
        for object in objects_in:
            self.objects[object[2]].append(object)
            if object[2] == DataClasses.BLUE:
                blue_count += 1
                center_sum = center_sum + np.array(object[0:2])
        if blue_count != 2:
            return False
        self.goal_center = center_sum/2
        return True
    
    def GenerateBallPath(self):
        path_points = np.array([self.objects[DataClasses.BALL][0][0:2]])
        problem_point = self.CheckColisions(np.array([self.objects[DataClasses.BALL][0][0:2], self.goal_center]))
        if problem_point is not None:
            path_points = self.GenerateWayAround(path_points[0], self.goal_center, problem_point)
        else:
            path_points = np.vstack([path_points, self.goal_center])
        return path_points
    
    def GenerateWayAround(self, start, end, problem):
        """Generates a path around the problem point by finding the shortest path through the tangent points."""
        tangent_point1, tangent_point2 = self.find_tangent_points(start, problem, PlanningParm.CLEARANCE)
        tangent_point3, tangent_point4 = self.find_tangent_points(end, problem, PlanningParm.CLEARANCE)
        combinations = [(tangent_point1, tangent_point3), (tangent_point1, tangent_point4), (tangent_point2, tangent_point3), (tangent_point2, tangent_point4)]
        shortest_path, min_distance = None, float('inf')
        for tangent_point_A, tangent_point_B in combinations:
            k1, q1 = self.line_equation(start, tangent_point_A)
            k2, q2 = self.line_equation(end, tangent_point_B)
            intersection = self.find_intersection(k1, q1, k2, q2)
            distance = self.path_length([start, tangent_point_A, intersection, tangent_point_B, end])
            if distance < min_distance:
                min_distance = distance
                shortest_path = (tangent_point_A, intersection, tangent_point_B)
        return np.array([start, np.array(shortest_path[1]), end])
    
    def GenerateShootPath(self):
        path_points =np.empty(2)
        shooting_point = self.ball_path[0]
        end_point = self.ball_path[1]
        direction = end_point - shooting_point
        norm_direction = direction / np.linalg.norm(direction)
        robot_start = shooting_point - norm_direction * PlanningParm.SHOOT_STEPBACK
        path_points = np.vstack([path_points, robot_start])
        shoot_point = robot_start + norm_direction * PlanningParm.SHOOT_SCALING
        path_points = np.vstack([path_points, shoot_point])
        return path_points
    
    def CheckColisions(self, path):
        colision_point = None
        for tube in self.objects[DataClasses.GREEN]:
            tube_pos = np.array(tube[0:2])
            distance = self.point_distance_from_line(tube_pos, path[0], path[1])
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
    
