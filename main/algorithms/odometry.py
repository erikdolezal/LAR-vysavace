class Odometry:
    """
    A class to manage and track the odometry of the robot.
    Attributes:
        turtle: Instance of the turtle_bot class.
        last_odometry: The last recorded odometry data of the turtle robot.
    Methods:
        __init__(turtle):
            Initializes the Odometry object with the given turtle robot and stores its initial odometry data.
        update_and_get_delta():
            Updates the stored odometry data and returns the previous and current odometry values as a tuple.
    """

    def __init__(self, turtle):
        self.turtle = turtle
        self.last_odometry = turtle.get_odometry()

    def update_and_get_delta(self):
        """
        Updates the stored odometry data and calculates the delta between the
        previous and current odometry readings.
        Returns:
            tuple: A tuple containing two elements:
                - odo_old: The previous odometry reading.
                - odo_new: The current odometry reading.
        """

        odo_new = self.turtle.get_odometry()
        odo_old = self.last_odometry
        self.last_odometry = odo_new
        return odo_old, odo_new