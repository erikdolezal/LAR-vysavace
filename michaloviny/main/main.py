from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
from camera import Camera
from UKFSlam import UKF_SLAM


class Odometry:
    def __init__(self, turtle):
        self.turtle = turtle
        self.last_odometry = turtle.get_odometry()

    def update_and_get_delta(self):
        odo_new = turtle.get_odometry()
        odo_old = self.last_odometry
        self.last_odometry = odo_new
        return odo_old, odo_new


if __name__ == "__main__":
    turtle = Turtlebot(rgb=True, depth=False, pc=True)
    camera = Camera(turtle)
    slam = UKF_SLAM(x_size=3, alpha=0.001, beta=2, kappa=0)
    odo = Odometry(turtle)
    while not turtle.is_shutting_down():
        objects = camera.get_np_objects()
        slam.update_from_detections(percep_data=objects)
        odo_old, odo_new = odo.update_and_get_delta()
        slam.predict(odo_old, odo_new)
