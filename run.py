from slam.UKFSlam import UKF_SLAM, DataClasses
from slam.geometry import local_to_global, global_to_local, rotate_points
from robolab_turtlebot import Turtlebot, sleep, Rate
from multiprocessing import Process, Queue, Event
from michaloviny.camera import Camera, OnnxCamera
from planning.PathPlanning import Planning
import numpy as np
import matplotlib.pyplot as plt
import time

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
        self.max_acc = 0.1 # m/s^2
        self.max_ang_acc = 0.1 # rad/s^2
        self.max_speed = 2 # m/s
        self.max_ang_speed = 4 # rad/s
        self.last_cmd = (0, 0)
        self.ang_p = 5
    
    def cmd_velocity(self, position, target_position, dt):
        target = global_to_local(np.expand_dims(target_position[:2].copy(), axis=0), position)[0]
        ang_velocity = np.arctan2(target[1], target[0])*self.ang_p
        self.velocity = target[0]
        if np.linalg.norm(target) < 0.1:
            self.velocity = 0
            ang_velocity = target_position[2] - position[2]
            #if abs(target_position[2] - position[2]) < np.deg2rad(0.3):
            #    ang_velocity = 0
        self.velocity = np.clip(np.clip(self.velocity, self.velocity - dt*self.max_acc, self.velocity + dt*self.max_acc), 
                                -self.max_speed, self.max_speed)
        ang_velocity = np.clip(ang_velocity, -self.max_ang_speed, self.max_ang_speed)
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
        self.velocity_control = VelocityControl(self.turtle)
        
        self.path_planning = Planning()

    
    def bumper_callback(self, msg):
        if msg.state == 1:
            self.end_event.set()
            print("Bumper pressed, ending program")


    def run(self):
        self.turtle.register_bumper_event_cb(self.bumper_callback)
        camera = OnnxCamera("michaloviny/best_ones/v11n_120e_160p.onnx", verbose=False, cam_K=self.turtle.get_rgb_K(), depth_K=self.turtle.get_depth_K(), conf_thresh=0.35)
        slam = UKF_SLAM(x_size=3, alpha=0.001, beta=2, kappa=0)
        odo = Odometry(self.turtle)
        slam_poses = np.zeros((0,3))
        rate = Rate(FREQUENCY)
        target = np.array([1, -0.1,0])
        ball = target
        last_time = time.perf_counter()
        while not self.turtle.is_shutting_down():
            if self.end_event.is_set():
                break
            st = time.perf_counter()
            img = self.turtle.get_rgb_image()
            depth_img = self.turtle.get_depth_image()
            objects = camera.get_detections(img, depth_img)
            print(f"camera time {(time.perf_counter() - st)*1000:.1f} ms")
            #print(objects[objects[:,2] == 3])
            if np.any(objects[:,2] == 3):
                ball[:2] = local_to_global(objects[objects[:,2] == 3, :2].copy(), slam.x[:3])[0]
            st = time.perf_counter()

            odo_old, odo_new = odo.update_and_get_delta()
            print(odo_old, odo_new)
            slam.predict(odo_new, odo_old)
            if objects.shape[0] > 0:
                #print(objects)
                #objects = objects[objects[:,2] == DataClasses.RED]
                objects = objects[objects[:,2] > 0.1]
                slam.update_from_detections(percep_data=objects)
            print(f"slam time {(time.perf_counter() - st)*1000:.1f} ms")
            slam_poses = np.vstack((slam_poses, slam.x[:3]))
            # timedelta calc
            actual_time = time.perf_counter()
            timedelta = actual_time - last_time
            last_time = actual_time
            pos_robot = np.append(slam.x[:2], [4])
            print(np.vstack([slam.landmarks, pos_robot]))
            point_togo = self.path_planning.CreatPath(np.vstack([slam.landmarks, pos_robot]), test_alg=False)
            print(point_togo)
            v_lin, v_ang = self.velocity_control.cmd_velocity(slam.x[:3], point_togo, timedelta)
            if np.linalg.norm(slam.x[:3] - ball) < 0.0:
                print("mission end")
                break
            print(f"velocity {v_lin} {v_ang}")
            self.turtle.cmd_velocity(v_lin, v_ang)
            rate.sleep()
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
        ax.plot(slam_poses[:,0], slam_poses[:,1], label="SLAM")
        ax.plot(slam_poses[-1,0], slam_poses[-1,1], 'go', label="End")
        blue_mask = slam.landmarks[:,2] == DataClasses.BLUE
        ax.plot(slam.landmarks[blue_mask,0], slam.landmarks[blue_mask,1], '.', c="blue", label="Blue cones")
        green_mask = slam.landmarks[:,2] == DataClasses.GREEN
        ax.plot(slam.landmarks[green_mask,0], slam.landmarks[green_mask,1], '.', c="green", label="Green cones")
        red_mask = slam.landmarks[:,2] == DataClasses.RED
        ax.plot(slam.landmarks[red_mask,0], slam.landmarks[red_mask,1], '.', c="red", label="Red cones")
        ax.legend()
        ax.grid()
        plt.show()


if __name__ == "__main__":
    control = MainControl()
    control.run()