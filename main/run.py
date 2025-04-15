from slam.UKFSlam import UKF_SLAM, DataClasses
from slam.geometry import local_to_global, global_to_local
from robolab_turtlebot import Turtlebot, Rate
from multiprocessing import Event
from camera.camera import OnnxCamera
from planning.PathPlanning import Planning
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

FREQUENCY = 10

class Odometry:
    def __init__(self, turtle):
        self.turtle = turtle
        self.last_odometry = turtle.get_odometry()

    def update_and_get_delta(self):
        odo_new = self.turtle.get_odometry()
        odo_old = self.last_odometry
        self.last_odometry = odo_new
        return odo_old, odo_new
    
class VelocityControl:
    def __init__(self, turtle):
        self.turtle = turtle
        self.velocity = 0
        self.max_acc = 0.8 # m/s^2
        self.max_ang_acc = 0.5 # rad/s^2
        self.max_speed = 0.8 # m/s #1.5
        self.max_ang_speed = 0.8 # rad/s
        self.last_cmd = (0, 0)
        self.ang_p = 1.5
    
    def cmd_velocity(self, position, target_position, dt):
        target = global_to_local(np.expand_dims(target_position[:2].copy(), axis=0), position)[0]
        ang_velocity = np.arctan2(target[1], target[0])*self.ang_p
        ang_velocity = np.clip(ang_velocity, -self.max_ang_speed, self.max_ang_speed)
        velocity = target[0] * (target[0] > 0) * (1 - (ang_velocity / self.max_ang_speed)**2) 
        self.velocity = np.clip(np.clip(velocity, self.velocity - dt*self.max_acc, self.velocity + dt*self.max_acc), 
                                -self.max_speed, self.max_speed)
        return self.velocity, ang_velocity
    
class MainControl:
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
        cv2.namedWindow("slam")
        cv2.resizeWindow("slam", 512, 512)
    
    def bumper_callback(self, msg):
        if msg.state == 1:
            self.end_event.set()
            print("Bumper pressed, ending program")

    def button_callback(self, msg):
        if msg.state == 1 and not self.start_event.is_set():
            self.start_event.set()
            print("Button pressed, starting mission")

    def run(self):
        self.turtle.register_bumper_event_cb(self.bumper_callback)
        self.turtle.register_button_event_cb(self.button_callback)
        print("waiting for start")
        slam_poses = np.zeros((1,3))
        rate = Rate(FREQUENCY)
        points = np.zeros((1,2))
        target = np.array([1, -0.1,0])
        ball = np.zeros((0,3))
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
            print(f"camera time {(time.perf_counter() - st)*1000:.1f} ms")
            #print(objects[objects[:,2] == 3])
            st = time.perf_counter()

            odo_old, odo_new = self.odo.update_and_get_delta()
            print(odo_old, odo_new)
            self.slam.predict(odo_new, odo_old)
            print(f"objects shape {objects.shape[0]}")
            if objects.shape[0] > 0 and (objects[objects[:,2] == DataClasses.BALL, :2].shape[0] == 0 or np.linalg.norm(objects[objects[:,2] == DataClasses.BALL, :2]) > 0.5):
                self.slam.update_from_detections(objects, st)
            print("slam pose", self.slam.x[:3])
            print(f"slam time {(time.perf_counter() - st)*1000:.1f} ms")
            if np.any(objects[:,2] == 3):
                ball = objects[objects[:,2] == DataClasses.BALL, :3][0]
                ball[:2] = local_to_global(objects[objects[:,2] == 3, :2].copy(), self.slam.x[:3])[0]
            slam_poses = np.vstack((slam_poses, self.slam.x[:3]))
            # timedelta calc
            actual_time = time.perf_counter()
            timedelta = actual_time - last_time
            last_time = actual_time
            #pos_robot = np.append(self.slam.x[:2], [4])
            print(np.vstack([self.slam.landmarks]))
            point_togo = self.path_planning.create_path(np.vstack([self.slam.landmarks, ball]), self.slam.x[:3], test_alg=False)
            print("togo", point_togo)
            if point_togo is None:
                print("goal")
                self.turtle.play_sound(sound_id=1)
                break
            else:
                v_lin, v_ang = self.velocity_control.cmd_velocity(self.slam.x[:3], point_togo, timedelta)
                #v_lin, v_ang = self.velocity_control.cmd_velocity(self.slam.x[:3], ball, timedelta)
                v_lin = 0 if self.slam.landmarks.shape[0] == 0 else v_lin
            points = np.vstack((points, point_togo))
            print(f"velocity {v_lin} {v_ang}")
            self.turtle.cmd_velocity(v_lin, v_ang)

            if self.end_event.is_set():
                if (objects[objects[:,2] == DataClasses.BALL, :2].shape[0] == 0 or np.linalg.norm(objects[objects[:,2] == DataClasses.BALL, :2]) > 0.5):
                    self.turtle.play_sound(sound_id=4)
                    break
                else:
                    self.end_event.clear()

            slam_win = np.ones((512,512,3), dtype=np.uint8) * 255
            cv2.circle(slam_win, (256, 256), 50, (0,0,0), 1)
            cv2.circle(slam_win, (256, 256), 100, (0,0,0), 1)
            cv2.circle(slam_win, (256, 256), 150, (0,0,0), 1)
            cv2.circle(slam_win, (256, 256), 200, (0,0,0), 1)
            cv2.circle(slam_win, (256, 256), 250, (0,0,0), 1)
            for landmark in self.slam.landmarks:
                if landmark[2] == DataClasses.BLUE:
                    color = (255, 0, 0)
                elif landmark[2] == DataClasses.GREEN:
                    color = (0, 255, 0)
                elif landmark[2] == DataClasses.RED:
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                cv2.circle(slam_win, (int(-landmark[1]*50) + 256, int(-landmark[0]*50) + 256), 3, color, -1)
            cv2.circle(slam_win, (int(-self.slam.x[1]*50) + 256, int(-self.slam.x[0]*50) + 256), 9, (0, 0, 0), -1)
            if ball.shape[0] > 0:
                cv2.circle(slam_win, (int(-ball[1]*50) + 256, int(-ball[0]*50) + 256), 5, (0, 255, 255), -1)
            cv2.line(slam_win, (int(-self.slam.x[1]*50) + 256, int(-self.slam.x[0]*50) + 256), (int(-point_togo[1]*50) + 256, int(-point_togo[0]*50) + 256), (255,0,255), 2)
            cv2.imshow("slam", slam_win)
            cv2.waitKey(1)

            #rate.sleep()
        self.turtle.cmd_velocity(0,0)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_title("SLAM")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.plot(slam_poses[0,0], slam_poses[0,1], 'ro', label="Start")
        ax.plot(slam_poses[-1,0], slam_poses[-1,1], 'go', label="End")
        ax.plot(slam_poses[:,0], slam_poses[:,1], '.', label="SLAM", c='orange')
        ax.plot(*points.T, c='violet', label='path')
        blue_mask = self.slam.landmarks[:,2] == DataClasses.BLUE
        ax.plot(self.slam.landmarks[blue_mask,0], self.slam.landmarks[blue_mask,1], '.', c="blue", label="Blue cones")
        green_mask = self.slam.landmarks[:,2] == DataClasses.GREEN
        ax.plot(self.slam.landmarks[green_mask,0], self.slam.landmarks[green_mask,1], '.', c="green", label="Green cones")
        red_mask = self.slam.landmarks[:,2] == DataClasses.RED
        ax.plot(self.slam.landmarks[red_mask,0], self.slam.landmarks[red_mask,1], '.', c="red", label="Red cones")
        ax.plot(*ball[:2], '.', label="Ball", c='cyan')
        ax.legend()
        ax.grid()
        plt.show()


if __name__ == "__main__":
    control = MainControl()
    control.run()