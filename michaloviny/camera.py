from __future__ import print_function

from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
from ultralytics import YOLO
import cv2
import time


SHOW = False

CAMERA_ANGLE = 10
WINDOW = "image"
MODEL_PATH = "michaloviny/best_ones/v11n_120e_160p.pt"
HALF_COORD_BOX = 2
R_PILLAR = 0.02
R_BALL = 0.08


label_map = {"green": 0, "red": 1, "blue": 2, "ball_y": 3, "ball_r": 3}


class ObjectData:
    def __init__(
        self,
        label: str,
        lower_left,
        width: int,
        height: int,
        confidence: int,
        coords: np.ndarray,
    ) -> None:
        self.label = label
        self.lower_left = lower_left
        self.width = width
        self.height = height
        self.camera_confidence = confidence
        self.coords = coords

    def __repr__(self) -> str:
        x = self.coords[0]
        y = self.coords[1]
        z = self.coords[2]
        return f"{self.label}: Pos (x: {x:.2f}, y: {y:.2f}, z: {z:.2f})m"


class Camera:
    def __init__(self, turtle):
        self.turtle = turtle
        self.R_x = self.get_correction_matrix(CAMERA_ANGLE)
        self.model = self.load_yolo_model()
        if SHOW:
            cv2.namedWindow(WINDOW)

    def generate_anotation(self, results, detected_objects) -> np.ndarray:
        """
        Generates anotations on image.
        """
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        for result in results:
            img = result.plot()

        for detected_object in detected_objects:
            x, y = detected_object.lower_left
            x_pos = detected_object.coords[0]
            y_pos = detected_object.coords[1]
            cv2.putText(
                img,
                f"({x_pos:.2f}, {y_pos:.2f}) m",
                (int(x), int(y - detected_object.height) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
        return img

    def adjust_coords(self, label, coords):
        """
        Adjust coords to match the center of an object.
        """
        if coords[0] == np.nan:
            return coords
        if "ball" in label:
            angle_z = np.arcsin(np.abs(coords[2]) / np.linalg.norm(coords))
            r_ball = R_BALL * np.cos(angle_z)
            coords += np.array([r_ball, r_ball, 0])
        else:
            coords += np.array([R_PILLAR, R_PILLAR, 0])
        return coords

    def get_coords(
        self, label, point_cloud: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """
        Get the coordinates of the object within a specified region of a point cloud.
        """
        y_middle = int((y1 + y2) // 2)
        x_middle = int((x1 + x2) // 2)
        region = point_cloud[
            y_middle - HALF_COORD_BOX : y_middle + HALF_COORD_BOX,
            x_middle - HALF_COORD_BOX : x_middle + HALF_COORD_BOX,
            :,
        ]
        median_coords = np.median(region, axis=(0, 1))
        # from xzy to xyz (xy is ground, z how tall)
        median_coords = median_coords[[0, 2, 1]]
        coords = self.R_x @ median_coords
        return self.adjust_coords(label, coords)

    def get_correction_matrix(self, angle_degrees):
        """
        Generates a correction matrix to adjust for a given camera tilt angle.
        """
        theta = np.radians(angle_degrees)
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        return R_x

    def load_yolo_model(self, model_path=MODEL_PATH) -> YOLO:
        return YOLO(model_path)

    def detect_objects(self):
        """
        Processes images from a Turtlebot's camera and Returns array of detected Objects with their coords.
        """
        st = time.perf_counter()
        #self.turtle.wait_for_rgb_image()
        img_rgb = self.turtle.get_rgb_image()
        depth_image = self.turtle.get_depth_image().copy()
        rgb_k = np.linalg.inv(self.turtle.get_rgb_K())
        depth_k = self.turtle.get_depth_K()
        #point_cloud = self.turtle.get_point_cloud()
        if img_rgb is None or depth_image is None:
            return []
        results = self.model(img_rgb)
        detected_objects = []
        K = depth_k @ rgb_k
        #print(K)
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, confidence, cls = box.tolist()
                if confidence < 0.3:
                    continue
                hom_coords = np.array([[(x1 + x2)/2, (y1 + y2)/2, 1]])
                world_coords = hom_coords @ rgb_k.T
                #print(hom_cords @ rgb_k.T)
                pic_coords = np.array([[x1, y1, 1], [x2, y2, 1]])
                label = result.names[int(cls)]
                lower_left = int(x1), int(y2)
                height = int(y2 - y1)
                width = int(x2 - x1)
                depth_coords = pic_coords @ K.T
                median_distance = np.median(depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])])
                distance = np.average(depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])], weights = np.exp(-(depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])] - median_distance)**2/100))
                world_coords *= distance/np.linalg.norm(world_coords[0])

                depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])] = distance

                #print(depth_coords, world_coords)
                coords = self.adjust_coords(label, self.R_x @ world_coords[0, [0,2,1]]/1000)
                #coords = self.get_coords(label, point_cloud, x1, y1, x2, y2)
                #print(depth_image.shape, img_rgb.shape, distance, x1, x2, y1, y2)
                detected_objects.append(
                    ObjectData(label, lower_left, width, height, confidence, coords)
                )
        #print("detect time:", time.perf_counter() - st)
        if SHOW:
            img_cam = self.generate_anotation(results, detected_objects)
            cv2.imshow(WINDOW, depth_image)
            cv2.waitKey(1)
        return detected_objects

    def get_np_objects(self):
        """
        Detect objects and parse them into numpy array.
        """
        objects = self.detect_objects()
        data = []
        for obj in objects:
            num_label = label_map.get(obj.label, -1)
            if num_label != -1:
                y, x = obj.coords[1], obj.coords[0]
                data.append([y, -x, num_label])
        return np.array(data)


if __name__ == "__main__":
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    camera = Camera(turtle)
    while not turtle.is_shutting_down():
        detected_objects = camera.get_np_objects()
        for detected_object in detected_objects:
            print(detected_object)
        print("-" * 20)
