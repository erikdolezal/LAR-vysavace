import numpy as np
from algorithms.geometry import global_to_local
from configs.alg_config import velocity_control_config


class VelocityControl:
    """
    A class to control the velocity of the robot, ensuring smooth acceleration
    and deceleration while adhering to specified constraints on speed and angular velocity.
    Attributes:
        turtle: Instance of the turtle_bot class.
        velocity (float): The current linear velocity of the turtle.
        max_acc (float): The maximum linear acceleration (m/s^2).
        max_ang_acc (float): The maximum angular acceleration (rad/s^2).
        max_speed (float): The maximum linear speed (m/s).
        max_ang_speed (float): The maximum angular speed (rad/s).
        last_cmd (tuple): The last commanded velocity as a tuple (linear, angular).
        ang_p (float): The proportional gain for angular velocity control.
    Methods:
        cmd_velocity(position, target_position, dt):
            Computes the linear and angular velocity commands to move the turtle
            towards a target position while respecting acceleration and speed limits.
    """

    def __init__(self):
        self.velocity = 0
        self.max_acc = velocity_control_config["max_acc"]  # m/s^2
        self.max_speed = velocity_control_config["max_speed"]  # m/s
        self.max_ang_speed = velocity_control_config["max_ang_speed"]  # rad/s
        self.last_cmd = (0, 0)
        self.ang_p = velocity_control_config["ang_p"]

    def cmd_velocity(self, position, target_position, dt):
        """
        Calculate the linear and angular velocity required to move towards a target position.
        Args:
            position (numpy.ndarray): The current position and orientation of the object
                in the global frame.
            target_position (numpy.ndarray): The target position in the global frame.
            dt (float): The time step for velocity calculation.
        Returns:
            tuple: A tuple containing:
                - velocity (float): The linear velocity to move towards the target position.
                - ang_velocity (float): The angular velocity to rotate towards the target position.
        """

        target = global_to_local(
            np.expand_dims(target_position[:2].copy(), axis=0), position
        )[0]
        ang_velocity = np.arctan2(target[1], target[0]) * self.ang_p
        ang_velocity = np.clip(ang_velocity, -self.max_ang_speed, self.max_ang_speed)
        velocity = (
            target[0] * (target[0] > 0) * (1 - (ang_velocity / self.max_ang_speed) ** 2)
        )
        self.velocity = np.clip(
            np.clip(
                velocity,
                self.velocity - dt * self.max_acc,
                self.velocity + dt * self.max_acc,
            ),
            -self.max_speed,
            self.max_speed,
        )
        return self.velocity, ang_velocity
