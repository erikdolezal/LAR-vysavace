import numpy as np
from configs.alg_config import PlanningParm
from configs.value_enums import ErrorCodes, DataClasses


class Planning:
    def __init__(self):
        self.objects = {}
        for type_ in DataClasses:
            self.objects[type_] = []
        self.goal_targer = None
        self.ball_pos = None
        self.ball_path = None
        self.robot_pos = None
        self.time_to_shoot = False

    def create_path(self, object_pos, robot_pos, test_alg=False):
        """
        Returns next point in a path or robot path and ball path if test_alg is true
        Args:
            object_pos (list): List of objects positions and classes
            robot_pos (list): Position of the robot
            test_alg (bool): If True, returns all paths for testing
        Returns:
            np.ndarray: Next point in the path for the robot to follow
        """

        next_point = None
        # Objects identification
        self.clear_objects()
        self.robot_pos = robot_pos
        inden_outcome = self.identify_objects(object_pos)
        if inden_outcome == ErrorCodes.IS_GOAL_ERR:
            self.time_to_shoot = False
            if test_alg:
                return None, None, None  # Returns None if the ball is in the goal
            next_point = None
        elif inden_outcome == ErrorCodes.NO_BALL_ERR or inden_outcome == ErrorCodes.ZERO_BLUE_ERR:
            print("No ball or no blue tube")
            next_point = self.turn_robot_around(self.robot_pos[:2], self.robot_pos[2], PlanningParm.ROBOT_TURN_RADIUS)
            if test_alg:
                return np.array([self.robot_pos[:2], next_point]), np.empty((0, 2)), np.empty((0, 2))  # Returns only point for rotation
            return next_point
        else:
            self.ball_path = self.generate_trajectory(self.ball_pos, self.goal_targer)  # Generates a path for the ball to the goal targer
            if self.ball_path is None:
                return ErrorCodes.BALL_STUCK_ERR

            shoot_path = self.generate_shoot_path()  # Generates a shooting point for the robot and path to the ball
            if shoot_path is None:
                return ErrorCodes.NO_SHOOT_ERR

            robot_destination = None
            hading_v = shoot_path[1] - shoot_path[0]
            if self.are_points_in_proximity(self.robot_pos[:2], shoot_path[0] + PlanningParm.SHOOT_ALIGNMENT * hading_v) and self.same_hading(hading_v, self.robot_pos[2]):
                print("Time to shoot")
                robot_destination = shoot_path[1]
                self.time_to_shoot = True
            elif self.are_points_in_proximity(self.robot_pos[:2], shoot_path[0] + PlanningParm.SHOOT_ALIGNMENT * hading_v):
                print("Time to turn")
                robot_destination = self.turn_robot_around(self.robot_pos[:2], self.robot_pos[2], PlanningParm.ROBOT_TURN_RADIUS)
                self.time_to_shoot = True
            elif self.are_points_in_proximity(self.robot_pos[:2], shoot_path[0]):
                print("Time to turn point")
                robot_destination = shoot_path[0] + PlanningParm.SHOOT_ALIGNMENT * hading_v
                self.time_to_shoot = True
            elif self.are_points_in_proximity(self.robot_pos[:2], self.ball_pos, PlanningParm.BALL_PROXIMITY):
                self.time_to_shoot = False
                robot_destination = self.robot_pos[:2]
            else:
                print("Time to move")
                self.time_to_shoot = False
                robot_destination = shoot_path[0]

            robot_path_to_shoot = self.generate_trajectory(self.robot_pos[:2], robot_destination)  # Generates a path for the robot to the shooting point
            if robot_path_to_shoot is None:
                return ErrorCodes.NO_ROBOT_ERR

            if test_alg:
                return robot_path_to_shoot, shoot_path, self.ball_path  # Returns all paths for testing
            next_point = robot_path_to_shoot[1]  # Sets the next point for the robot to follow

        return next_point  # Returns the next point in the path for the robot to follow

    def clear_objects(self):
        """
        Clears all objectss.
        """

        for type_ in DataClasses:
            self.objects[type_] = []

    def same_hading(self, vector, angle):
        """
        Checks if the robot is in the same direction as the vector.
        Args:
            vector (np.ndarray): Vector to check
            angle (float): Angle of the robot
        Returns:
            bool: True if the robot is in the same direction as the vector, False otherwise
        """

        aplha = np.arctan2(vector[1], vector[0])
        if np.abs(aplha - angle) < PlanningParm.HADING_CHECK:
            return True
        return False

    def turn_robot_around(self, center, angle, distance):
        """
        Turns the robot around by -90 degrees.
        Args:
            center (np.ndarray): Center of the robot
            angle (float): Angle of the robot
            distance (float): Scale of turning vector
        Returns:
            np.ndarray: New position to turn to.
        """

        vector = np.array([distance * np.cos(angle), distance * np.sin(angle)])
        vector = self.mat_rot(-np.pi / 2, vector)
        vector = vector / np.linalg.norm(vector) * distance
        return center + vector

    def identify_objects(self, objects_in):
        """
        Indenfify objects and add them to the objects list
        Args:
            objects_in (np.ndarray): List of objects positions and classes
        Returns:
            ErrorCodes: Error code for the outcome of the function
        """

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

        if poss_balls.size != 0:
            self.ball_pos = np.mean(poss_balls, axis=0)  # Calculates the mean of all ball positions if the possition is not sure
        if self.ball_pos is None:
            return ErrorCodes.NO_BALL_ERR

        if blue_count > 2:
            self.goal_targer = self.solve_more_blue_tubes()
        elif blue_count == 2:
            self.goal_targer = center_sum / 2  # Calculates the center of the goal
            if self.is_goal():
                return ErrorCodes.IS_GOAL_ERR
        elif blue_count == 1 and self.goal_targer is not None:
            self.goal_targer = self.goal_targer
        elif blue_count == 1:
            return ErrorCodes.ZERO_BLUE_ERR
        else:
            if self.goal_targer is None:
                return ErrorCodes.ZERO_BLUE_ERR
            else:
                # check if is goal
                return ErrorCodes.OK_ERR
        return ErrorCodes.OK_ERR

    def solve_more_blue_tubes(self):
        """
        Solves the problem of more than 2 blue tubes by returning the first blue tube or the goal target if it exists
        Retuns:
            np.ndarray: Position of the blue tube or goal target
        """

        blue_tubes = self.objects[DataClasses.BLUE]
        if self.goal_targer is None:
            return blue_tubes[0]
        return self.goal_targer

    def is_goal(self):
        """
        Checks if the ball is in the goal.
        Returns:
            bool: True if the ball is in the goal, False otherwise
        """

        if self.ball_pos is not None and self.goal_targer is not None:
            # vec_behind = self.goal_targer - self.objects[DataClasses.BLUE][0][0:2]
            # ver_beh_rot = self.mat_rot(-np.pi/2, vec_behind)
            blue_vec = self.objects[DataClasses.BLUE][1][0:2] - self.objects[DataClasses.BLUE][0][0:2]
            t = np.dot(self.robot_pos[:2] - self.objects[DataClasses.BLUE][0][0:2], blue_vec) / np.linalg.norm(blue_vec) ** 2
            t = np.clip(t, 0, 1)
            robot_dist = np.linalg.norm(self.robot_pos[:2] - (self.objects[DataClasses.BLUE][0][0:2] + t * blue_vec))
            behind_goal = self.goal_targer - self.robot_pos[:2]
            projection = (np.dot(blue_vec, behind_goal) / np.dot(blue_vec, blue_vec)) * blue_vec
            behind_goal = behind_goal - projection
            behind_goal = (behind_goal / np.linalg.norm(behind_goal) * PlanningParm.GOAL_CHECK)
            dis_to_target = np.linalg.norm(self.ball_pos - (self.goal_targer - behind_goal))
            dis_to_check = np.linalg.norm(self.ball_pos - (self.goal_targer + behind_goal))
            if dis_to_target > dis_to_check or robot_dist < PlanningParm.ROBOT_DIST_MIN:
                return True
        return False

    def generate_trajectory(self, start, target):
        """
        Gernerates a path from start to target point and avoids one obstacle point.
        Args:
            start (np.ndarray): Start point
            target (np.ndarray): Target point
        Returns:
            np.ndarray: Path points
        """

        path_points = None
        problem_point = self.check_colisions(np.array([start, target]))  # Checks for colision of a direct path between start and target

        if problem_point is not None:
            path_points = self.generate_way_around(start, target, problem_point)  # Generates a path around the problematic point
        else:
            path_points = np.array([start, target])  # Creates direct path to the target

        return path_points

    def generate_way_around(self, start, end, problem):
        """
        Generates a path around the problem point by finding the shortest path through the tangent points.
        Args:
            start (np.ndarray): Start point
            end (np.ndarray): End point
            problem (np.ndarray): Problem point
        Returns:
            np.ndarray: Path points for the alternative path
        """

        if np.linalg.norm(start - problem) < PlanningParm.CLEARANCE or np.linalg.norm(end - problem) < PlanningParm.CLEARANCE:
            return np.array([start, end])

        # Finds the tangent points from the start and end points to the problem point
        tangent_point1, tangent_point2 = self.find_tangent_points(start, problem, PlanningParm.CLEARANCE)
        tangent_point3, tangent_point4 = self.find_tangent_points(end, problem, PlanningParm.CLEARANCE)

        # Creates a list of all possible combinations of tangent points
        combinations = [(tangent_point1, tangent_point3), (tangent_point1, tangent_point4), (tangent_point2, tangent_point3), (tangent_point2, tangent_point4)]

        shortest_inter, min_distance = None, float('inf')
        for tangent_point_A, tangent_point_B in combinations:
            intersection = self.find_intersection(start, tangent_point_A, end, tangent_point_B)  # finds the intersection point of the two tangent lines
            distance = self.path_length([start, intersection, end])
            if distance < min_distance:
                min_distance = distance
                shortest_inter = intersection

        return np.array([start, shortest_inter, end])

    def generate_shoot_path(self):
        """
        Generates a shooting point for the robot and a path to the ball
        Returns:
            np.ndarray: Shooting point and overshoot point
        """

        ball_point = self.ball_path[0]
        end_point = self.ball_path[1]
        direction = end_point - ball_point
        norm_direction = direction / np.linalg.norm(direction)
        shooting_point = ball_point - norm_direction * PlanningParm.SHOOT_STEPBACK  # Calculates the shooting point by moving back from the ball point
        overshoot_point = ball_point + norm_direction * PlanningParm.SHOOT_SCALING  # Calculates the overshoot point to gain speed

        return np.array([shooting_point, overshoot_point])

    def check_colisions(self, path):
        """
        Checks if there are any collisions with the path between the start and end points and returns the collision point if there is one.
        Args:
            path (np.ndarray): Path points
        Returns:
            np.ndarray: Collision point if there is one, else None
        """

        colision_point = None
        for data_class in DataClasses:
            if data_class == DataClasses.BLUE:
                continue
            elif data_class == DataClasses.BALL and self.time_to_shoot:
                continue
            else:
                colision = self.check_colision_from_class(path, data_class)
                if colision is not None:
                    colision_point = colision
                    break

        return colision_point

    def check_colision_from_class(self, path, data_class):
        """
        Checks for colision with specified class of objects and returns the colision point if there is one.
        Args:
            path (np.ndarray): Path points
            data_class (DataClasses): Class of objects
        Returns:
            np.ndarray: Collision point if there is one, else None
        """

        colision = None
        for tube in self.objects[data_class]:  # Checks for colision with all green tubes
            tube_pos = np.array(tube[0:2])
            distance = self.point_distance_from_line(tube_pos, path[0], path[1])  # Calculates the distance from the line to the tube
            if distance < PlanningParm.CLEARANCE:
                colision = tube_pos
                break
        return colision

    def point_distance_from_line(self, P, A, B):
        """
        Calculates the perpendicular distance of a point from the line defined by points A and B.
        Args:
            P (np.ndarray): Point to check
            A (np.ndarray): Start point of the line
            B (np.ndarray): End point of the line
        Returns:
            float: Distance from the point to the line
        """

        AB = B - A
        AP = P - A

        # Projecton of AP onto AB
        AB_dot_AB = np.dot(AB, AB)
        AP_dot_AB = np.dot(AP, AB)

        t = AP_dot_AB / AB_dot_AB

        # If t is outside the segment, return the distance to the nearest endpoint
        if t < 0.0:
            nearest_point = A
        elif t > 1.0:
            nearest_point = B
        else:
            nearest_point = A + t * AB

        return np.linalg.norm(P - nearest_point)

    def find_tangent_points(self, start, center, radius):
        """
        Finds two tangent points from start point to circle.
        Args:
            start (np.ndarray): Start point
            center (np.ndarray): Center of the object
            radius (float): Radius of the object perimeter
        Returns:
            np.ndarray: First tangent point
            np.ndarray: Second tangent point
        """

        d = start - center
        d_norm = np.linalg.norm(d)
        angle = np.arccos(radius / d_norm)
        d_unit = d / d_norm

        dir1 = self.mat_rot(angle, d_unit)
        dir2 = self.mat_rot(-angle, d_unit)
        tangent_point1 = center + radius * dir1
        tangent_point2 = center + radius * dir2

        return tangent_point1, tangent_point2

    def mat_rot(self, angle, vector):
        """
        Returns a rotation matrix for a given angle
        Args:
            angle (float): Angle in radians
            vector (np.ndarray): Vector to rotate
        Returns:
            np.ndarray: Rotated vector
        """

        mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_vector = np.dot(mat, vector)
        return rotated_vector

    def path_length(self, points):
        """
        Calculates the total length of a line.
        Args:
            float: Length of the line.
        """

        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    def find_intersection(self, lineA_a, lineA_b, lineB_a, lineB_b):
        """
        Finds the intersection point of two lines defined by two points each.
        1. lineA: (lineA_a, lineA_b)
        2. lineB: (lineB_a, lineB_b)
        3. Returns the intersection point if exists, else None
        Args:
            lineA_a (np.ndarray): Start point of line A
            lineA_b (np.ndarray): End point of line A
            lineB_a (np.ndarray): Start point of line B
            lineB_b (np.ndarray): End point of line B
        Returns:
            np.ndarray: Intersection point if exists, else None
        """

        lineA = lineA_b - lineA_a
        lineB = lineB_b - lineB_a
        M = np.column_stack([lineA, -lineB])
        rhs = lineB_a - lineA_a

        if np.linalg.det(M) == 0:
            return None  # Lines are parallel and do not intersect

        t, s = np.linalg.solve(M, rhs)
        intersection = lineA_a + t * lineA
        return intersection

    def are_points_in_proximity(self, point1, point2, dis_check=PlanningParm.BALL_PROXIMITY):
        """
        Checks if two points are within a given proximity.
        Args:
            point1 (np.ndarray): First point
            point2 (np.ndarray): Second point
            dis_check (float): Distance threshold for proximity check
        Returns:
            bool: True if points are within proximity, False otherwise
        """
        state = False
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        if distance < dis_check:
            state = True
        return state
