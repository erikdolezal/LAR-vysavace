Module main.planning.PathPlanning
=================================

Classes
-------

`Planning()`
:   

    ### Methods

    `are_points_in_proximity(self, point1, point2, dis_check=0.2)`
    :   Checks if two points are within a given proximity.

    `check_colision_from_class(self, path, data_class)`
    :   Checks for colision with specified class of objects and returns the colision point if there is one.

    `check_colisions(self, path)`
    :   Checks if there are any collisions with the path between the start and end points and returns the collision point if there is one.

    `clear_objects(self)`
    :   Clears all objectss

    `create_path(self, object_pos, robot_pos, test_alg=False)`
    :   Returns next point in a path or robot path and ball path if test_alg is true

    `find_intersection(self, lineA_a, lineA_b, lineB_a, lineB_b)`
    :   Finds the intersection point of two lines defined by two points each.
        1. lineA: (lineA_a, lineA_b)
        2. lineB: (lineB_a, lineB_b)
        3. Returns the intersection point if exists, else None

    `find_tangent_points(self, start, center, radius)`
    :   Finds two tangent points from start point to circle.

    `generate_shoot_path(self)`
    :   Generates a shooting point for the robot and a path to the ball

    `generate_trajectory(self, start, target)`
    :   Gernerates a path from start to target point and avoids one obstacle point

    `generate_way_around(self, start, end, problem)`
    :   Generates a path around the problem point by finding the shortest path through the tangent points.

    `identify_objects(self, objects_in)`
    :   Indenfify objects and add them to the objects list

    `is_goal(self)`
    :   Checks if the ball is in the goal

    `mat_rot(self, angle, vector)`
    :   Returns a rotation matrix for a given angle

    `path_length(self, points)`
    :   Calculates the total length of a line.

    `point_distance_from_line(self, P, A, B)`
    :   Calculates the perpendicular distance of a point from the line defined by points A and B.

    `same_hading(self, vector, angle)`
    :   Checks if the robot is in the same direction as the vector

    `solve_more_blue_tubes(self)`
    :   Solves the problem of more than 2 blue tubes by returning the first blue tube or the goal target if it exists

    `turn_robot_around(self, center, angle, distance)`
    :   Turns the robot around by -90 degrees