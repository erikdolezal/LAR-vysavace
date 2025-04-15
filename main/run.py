from algorithms.UKFSlam import UKF_SLAM, DataClasses
from algorithms.geometry import local_to_global, global_to_local
from robolab_turtlebot import Turtlebot
from multiprocessing import Event
from algorithms.camera import OnnxCamera
from algorithms.PathPlanning import Planning
from algorithms.odometry import Odometry
from algorithms.velocity_control import VelocityControl
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

class MainControl:
    """
    MainControl class is responsible for managing the robot's main control loop,
    handling  SLAM (Simultaneous Localization and Mapping),
    path planning, and velocity control.
    Attributes:
        turtle (Turtlebot): Instance of the Turtlebot class for robot control.
        end_event (Event): Event to signal the end of the program.
        start_event (Event): Event to signal the start of the mission.
        velocity_control (VelocityControl): Handles velocity commands for the robot.
        camera (OnnxCamera): Object detection class.
        slam (UKF_SLAM): SLAM classs.
        odo (Odometry): Odometry class.
        path_planning (Planning): Path planning  class for navigation.
    Methods:
        __init__():
            Initializes the MainControl class, sets up the robot, SLAM, and other components.
        bumper_callback(msg):
            Callback function triggered when the bumper is pressed.
        button_callback(msg):
            Callback function triggered when the button is pressed.
        run():
            Main control loop for the robot. Handles sensor data, SLAM updates,
            path planning, and robot motion.
    """

    def __init__(self):
        self.turtle = Turtlebot(rgb=True, depth=True, pc=True)
        self.turtle.reset_odometry()
        while not self.turtle.has_rgb_image() and not self.turtle.has_depth_image():
            time.sleep(0.1)
        print(f"Has RGB Image: {self.turtle.has_rgb_image()}")
        print(f"Has Depth Image: {self.turtle.has_depth_image()}")

        self.turtle.get_rgb_image()
        self.turtle.get_depth_image()

        self.end_event = Event()
        self.start_event = Event()
        self.velocity_control = VelocityControl(self.turtle)
        self.camera = OnnxCamera("yolo/v11s_v2_300e_160p.onnx", verbose=False, cam_K=self.turtle.get_rgb_K(), depth_K=self.turtle.get_depth_K(), conf_thresh=0.30)
        self.slam = UKF_SLAM(x_size=3, alpha=0.001, beta=2, kappa=0)
        self.odo = Odometry(self.turtle)
        self.path_planning = Planning()
        self.slam_poses = np.zeros((1, 3))
        self.points = np.zeros((1, 2))
        self.ball = np.zeros((0, 3))
        cv2.namedWindow("slam")
        cv2.resizeWindow("slam", 512, 512)

    def bumper_callback(self, msg):
        """
        Callback function triggered when the bumper state changes.
        Args:
            msg: A message object containing the state of the bumper.
        """
        if msg.state == 1:
            self.end_event.set()
            print("Bumper pressed, ending program")

    def button_callback(self, msg):
        """
        Callback function triggered by a button press event used for starting the mission.
        Args:
            msg: An object containing the state of the button.
        """

        if msg.state == 1 and not self.start_event.is_set():
            self.start_event.set()
            print("Button pressed, starting mission")

    def run(self):
        """
        The `run` method is the main execution loop for the robot's operation. It handles sensor data processing,
        SLAM (Simultaneous Localization and Mapping), path planning, and velocity control. The method also
        visualizes the SLAM process and generates a final plot of the robot's path and detected landmarks.
        The method continuously runs until the robot is shut down or a goal is reached.
        """

        self.turtle.register_bumper_event_cb(self.bumper_callback)
        self.turtle.register_button_event_cb(self.button_callback)
        print("waiting for start")
        last_time = time.perf_counter()
        self.turtle.play_sound(sound_id=0)
        while not self.turtle.is_shutting_down():
            if not self.start_event.is_set():
                time.sleep(0.1)
                continue
            st = time.perf_counter()
            img = self.turtle.get_rgb_image()
            depth_img = self.turtle.get_depth_image()
            objects = self.camera.get_detections(img, depth_img)
            print(f"camera time {(time.perf_counter() - st) * 1000:.1f} ms")
            st = time.perf_counter()

            odo_old, odo_new = self.odo.update_and_get_delta()
            print(odo_old, odo_new)
            self.slam.predict(odo_new, odo_old)
            print(f"objects shape {objects.shape[0]}")
            if objects.shape[0] > 0 and (
                objects[objects[:, 2] == DataClasses.BALL, :2].shape[0] == 0
                or np.linalg.norm(objects[objects[:, 2] == DataClasses.BALL, :2]) > 0.5
            ):
                self.slam.update_from_detections(objects, st)
            print("slam pose", self.slam.x[:3])
            print(f"slam time {(time.perf_counter() - st) * 1000:.1f} ms")
            if np.any(objects[:, 2] == 3):
                self.ball = objects[objects[:, 2] == DataClasses.BALL, :3][0]
                self.ball[:2] = local_to_global(
                    objects[objects[:, 2] == 3, :2].copy(), self.slam.x[:3]
                )[0]
            self.slam_poses = np.vstack((self.slam_poses, self.slam.x[:3]))
            # timedelta calc
            actual_time = time.perf_counter()
            timedelta = actual_time - last_time
            last_time = actual_time
            print(np.vstack([self.slam.landmarks]))
            point_togo = self.path_planning.create_path(
                np.vstack([self.slam.landmarks, self.ball]), self.slam.x[:3], test_alg=False
            )
            print("togo", point_togo)
            if point_togo is None:
                print("goal")
                self.turtle.play_sound(sound_id=1)
                break
            else:
                v_lin, v_ang = self.velocity_control.cmd_velocity(
                    self.slam.x[:3], point_togo, timedelta
                )
                v_lin = 0 if self.slam.landmarks.shape[0] == 0 else v_lin
            self.points = np.vstack((self.points, point_togo))
            print(f"velocity {v_lin} {v_ang}")
            self.turtle.cmd_velocity(v_lin, v_ang)

            if self.end_event.is_set():
                if (
                    objects[objects[:, 2] == DataClasses.BALL, :2].shape[0] == 0
                    or np.linalg.norm(objects[objects[:, 2] == DataClasses.BALL, :2])
                    > 0.5
                ):
                    self.turtle.play_sound(sound_id=4)
                    break
                else:
                    self.end_event.clear()

            slam_win = np.ones((512, 512, 3), dtype=np.uint8) * 255
            cv2.circle(slam_win, (256, 256), 50, (0, 0, 0), 1)
            cv2.circle(slam_win, (256, 256), 100, (0, 0, 0), 1)
            cv2.circle(slam_win, (256, 256), 150, (0, 0, 0), 1)
            cv2.circle(slam_win, (256, 256), 200, (0, 0, 0), 1)
            cv2.circle(slam_win, (256, 256), 250, (0, 0, 0), 1)
            for landmark in self.slam.landmarks:
                if landmark[2] == DataClasses.BLUE:
                    color = (255, 0, 0)
                elif landmark[2] == DataClasses.GREEN:
                    color = (0, 255, 0)
                elif landmark[2] == DataClasses.RED:
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                cv2.circle(
                    slam_win,
                    (int(-landmark[1] * 50) + 256, int(-landmark[0] * 50) + 256),
                    3,
                    color,
                    -1,
                )
            cv2.circle(
                slam_win,
                (int(-self.slam.x[1] * 50) + 256, int(-self.slam.x[0] * 50) + 256),
                9,
                (0, 0, 0),
                -1,
            )
            if self.ball.shape[0] > 0:
                cv2.circle(
                    slam_win,
                    (int(-self.ball[1] * 50) + 256, int(-self.ball[0] * 50) + 256),
                    5,
                    (0, 255, 255),
                    -1,
                )
            cv2.line(
                slam_win,
                (int(-self.slam.x[1] * 50) + 256, int(-self.slam.x[0] * 50) + 256),
                (int(-point_togo[1] * 50) + 256, int(-point_togo[0] * 50) + 256),
                (255, 0, 255),
                2,
            )
            cv2.imshow("slam", slam_win)
            cv2.waitKey(1)

        self.turtle.cmd_velocity(0, 0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_title("SLAM")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.plot(self.slam_poses[0,0], self.slam_poses[0,1], 'ro', label="Start")
        ax.plot(self.slam_poses[-1,0], self.slam_poses[-1,1], 'go', label="End")
        ax.plot(self.slam_poses[:,0], self.slam_poses[:,1], '.', label="SLAM", c='orange')
        ax.plot(*self.points.T, c='violet', label='path')
        blue_mask = self.slam.landmarks[:,2] == DataClasses.BLUE
        ax.plot(self.slam.landmarks[blue_mask,0], self.slam.landmarks[blue_mask,1], '.', c="blue", label="Blue cones")
        green_mask = self.slam.landmarks[:,2] == DataClasses.GREEN
        ax.plot(self.slam.landmarks[green_mask,0], self.slam.landmarks[green_mask,1], '.', c="green", label="Green cones")
        red_mask = self.slam.landmarks[:,2] == DataClasses.RED
        ax.plot(self.slam.landmarks[red_mask,0], self.slam.landmarks[red_mask,1], '.', c="red", label="Red cones")
        ax.plot(*self.ball[:2], '.', label="ball", c='cyan')
        ax.legend()
        ax.grid()
        plt.show()


if __name__ == "__main__":
    control = MainControl()
    control.run()
