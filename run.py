
from slam.UKFSlam import UKF_Slam
from slam.geometry import local_to_global, global_to_local, rotate_points
from robolab_turtlebot import Turtlebot, sleep, Rate
from multiprocessing import Process, Queue, Event
from michaloviny.camera import Camera
import numpy as np

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
        self.max_speed = 1 # m/s
        self.max_ang_speed = 1 # rad/s
        self.last_cmd = (0, 0)
    
    def cmd_velocity(self, position, target_position):
        # compute distance
        distance = np.linalg.norm(target_position - position)
        # compute angle
        angle = np.arctan2(target_position[1] - position[1], target_position[0] - position[0])
        # compute angle difference
        angle_diff = angle - position[2]
        # compute velocity
        velocity = np.clip(distance, -self.max_speed, self.max_speed)
        # compute angular velocity
        ang_velocity = np.clip(angle_diff, -self.max_ang_speed, self.max_ang_speed)
        # compute acceleration
        acc = np.clip(velocity - self.last_cmd[0], -self.max_acc, self.max_acc)
        # compute angular acceleration
        ang_acc = np.clip(ang_velocity - self.last_cmd[1], -self.max_ang_acc, self.max_ang_acc)
        # store last command
        self.last_cmd = (velocity, ang_velocity)
        # return command
        return velocity, ang_velocity, acc, ang_acc
    
def camera_process(turtle, camera_queue, end_event):
    print("Camera process started")
    camera = Camera(turtle)
    while not turtle.is_shutting_down():
        if end_event.is_set():
            break
        objects = camera.get_np_objects()
        camera_queue.put(objects)
    print("Camera process ended")

def conrol_loop(turtle, camera_queue, end_event):
    print("Control process started")
    slam = UKF_Slam(x_size=3, alpha=0.001, beta=2, kappa=0)
    odo = Odometry(turtle)
    rate = Rate(FREQUENCY)
    while not turtle.is_shutting_down():
        if end_event.is_set():
            turtle.cmd_velocity(0, 0)
            break

        odo_old, odo_new = odo.update_and_get_delta()
        slam.predict(odo_old, odo_new)
        if not camera_queue.empty():
            objects = camera_queue.get()
            if objects.shape[0] > 0:
                slam.update_from_detections(percep_data=objects)
        rate.sleep()
    print("Control process ended")

class MainControl:
    def __init__(self):
        self.turtle = Turtlebot(rgb=True, depth=True, pc=True)
        self.end_event = Event
        # start camera process
        self.camera_queue = Queue()
        self.camera_process = Process(target=camera_process, args=(self.turtle, self.camera_queue, self.end_event))
        self.camera_process.start()
        # start control process
        self.control_process = Process(target=conrol_loop, args=(self.turtle, self.camera_queue, self.end_event))
        self.camera_process.start()
    
    def bumper_callback(self, msg):
        if msg.state == 1:
            self.end_event.set()
            print("Bumper pressed, ending program")
            exit(0)


    def run(self):
        self.turtle.register_bumper_event_cb(self.bumper_callback)
        self.camera_process.join()
        self.control_process.join()


if __name__ == "__main__":
    control = MainControl()
    control.run()