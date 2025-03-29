from slam.UKFSlam import UKF_SLAM, DataClasses
from slam.geometry import local_to_global, global_to_local, rotate_points
from robolab_turtlebot import Turtlebot, sleep, Rate
from multiprocessing import Process, Queue, Event
from michaloviny.camera import Camera
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
        self.max_acc = 0.1 # m/s^2
        self.max_ang_acc = 0.1 # rad/s^2
        self.max_speed = 0.1 # m/s
        self.max_ang_speed = 1 # rad/s
        self.last_cmd = (0, 0)
    
    def cmd_velocity(self, position, target_position):
        target = global_to_local(np.expand_dims(target_position[:2], axis=0), position)[0]
        ang_velocity = np.arctan2(target[1], target[0])
        velocity = target[0]
        if np.linalg.norm(target) < 0.1:
            velocity = 0
            ang_velocity = target_position[2] - position[2]
            if abs(target_position[2] - position[2]) < np.deg2rad(0.3):
                ang_velocity = 0
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        ang_velocity = np.clip(ang_velocity, -self.max_ang_speed, self.max_ang_speed)
        return velocity, ang_velocity#, acc, ang_acc
    
def camera_process(turtle, camera_queue, end_event):
    print("Camera process started")
    camera = Camera(turtle)
    while not turtle.is_shutting_down():
        if end_event.is_set():
            break
        st = time.perf_counter()
        objects = camera.get_np_objects()
        print(f"camera loop time {(time.perf_counter() - st)*1000:.1f} ms")
        camera_queue.put(objects)
    print("Camera process ended")

def conrol_loop(turtle, camera_queue, end_event):
    print("Control process started")
    slam = UKF_SLAM(x_size=3, alpha=0.001, beta=2, kappa=0)
    odo = Odometry(turtle)
    slam_poses = np.zeros((0,3))
    rate = Rate(FREQUENCY)
    while not turtle.is_shutting_down():
        if end_event.is_set():
            turtle.cmd_velocity(0, 0)
            break
        odo_old, odo_new = odo.update_and_get_delta()
        slam.predict(np.zeros(3), np.zeros(3))
        if not camera_queue.empty():
            objects = camera_queue.get()
            if objects.shape[0] > 0:
                print(objects)
                slam.update_from_detections(percep_data=objects)
        slam_poses = np.vstack((slam_poses, slam.x[:3]))
        rate.sleep()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_title("SLAM")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
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
    print("Control process ended")

class MainControl:
    def __init__(self):
        self.turtle = Turtlebot(rgb=True, depth=True, pc=True)
        self.turtle.reset_odometry()
        while not self.turtle.has_rgb_image() and not self.turtle.has_depth_image():
            time.sleep(0.1)
        print(f"Has RGB Image: {self.turtle.has_rgb_image()}")
        #print(self.turtle.get_rgb_image())
        print(f"Has Depth Image: {self.turtle.has_depth_image()}")
        #print(f"Has Point Cloud: {self.turtle.has_point_cloud()}")
        #print(self.turtle.get_point_cloud())
        self.turtle.get_rgb_image()
        self.turtle.get_depth_image()
        #exit(0)
        self.end_event = Event()
        self.velocity_control = VelocityControl(self.turtle)
        # start camera process
        #self.camera_queue = Queue()
        #self.camera_process = Process(target=camera_process, args=(self.turtle, self.camera_queue, self.end_event))
        #self.camera_process.start()
        # start control process
        #self.control_process = Process(target=conrol_loop, args=(self.turtle, self.camera_queue, self.end_event))
        #self.control_process.start()
    
    def bumper_callback(self, msg):
        if msg.state == 1:
            self.end_event.set()
            print("Bumper pressed, ending program")


    def run(self):
        self.turtle.register_bumper_event_cb(self.bumper_callback)
        camera = Camera(self.turtle)
        slam = UKF_SLAM(x_size=3, alpha=0.001, beta=2, kappa=0)
        odo = Odometry(self.turtle)
        slam_poses = np.zeros((0,3))
        rate = Rate(FREQUENCY)
        start = time.perf_counter()
        while not self.turtle.is_shutting_down():
            if self.end_event.is_set():
                break
            st = time.perf_counter()
            objects = camera.get_np_objects()
            print(f"camera time {(time.perf_counter() - st)*1000:.1f} ms")
            #self.camera_queue.put(objects)
            st = time.perf_counter()

            odo_old, odo_new = odo.update_and_get_delta()
            print(odo_old, odo_new)
            slam.predict(odo_new, odo_old)
            #if not camera_queue.empty():
            #    objects = camera_queue.get()
            if objects.shape[0] > 0:
                print(objects)
                objects = objects[objects[:,2] == DataClasses.RED]
                slam.update_from_detections(percep_data=objects)
            print(f"slam time {(time.perf_counter() - st)*1000:.1f} ms")
            slam_poses = np.vstack((slam_poses, slam.x[:3]))
            v_lin, v_ang = self.velocity_control.cmd_velocity(slam.x[:3], np.array([1,0.,0]))
            print(f"velocity {v_lin} {v_ang}")
            self.turtle.cmd_velocity(v_lin, v_ang)
            #if time.perf_counter() - start < 1:
            #else:
            #    self.turtle.cmd_velocity(0,0)
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
        #while not self.turtle.is_shutting_down():
        #    if self.end_event.is_set():
        #        break
        #if self.camera_process.is_alive():
        #    while not self.camera_queue.empty():
        #        self.camera_queue.get()
        #    self.camera_process.terminate()
        #if self.control_process.is_alive():
        #    self.control_process.terminate()


if __name__ == "__main__":
    control = MainControl()
    control.run()