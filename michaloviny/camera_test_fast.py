from __future__ import print_function

from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
from ultralytics import YOLO
import cv2


CAMERA_ANGLE = 20

WINDOW = "image"
MODEL_PATH = "best_ones/v11n_120e_160p.pt"
DISTANCE_PERCENTILE = 80
HALF_COORD_BOX = 2


class ObjectData:
    def __init__(
        self,
        label: str,
        lower_left: tuple[int, int],
        width: int,
        height: int,
        coords: np.ndarray,
    ) -> None:
        self.label = label
        self.lower_left = lower_left
        self.width = width
        self.height = height
        self.coords = coords

    def __repr__(self) -> str:
        x = self.coords[0]
        y = self.coords[1]
        z = self.coords[2]
        return f"{self.label}: Pos ({x:.2f}, {y:.2f}, {z:.2f})m"


def generate_anotation(results, detected_objects: list[ObjectData]) -> np.ndarray:
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
            f"c:({x_pos:.2f}, {y_pos:.2f})m",
            (int(x), int(y - detected_object.height) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
    return img


def get_coords(
    point_cloud: np.ndarray, x1: int, y1: int, x2: int, y2: int, R_x: np.ndarray
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

    coords = R_x @ median_coords

    return coords


def get_correction_matrix(angle_degrees):
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


def load_yolo_model(model_path=MODEL_PATH) -> YOLO:
    return YOLO(model_path)


def detect_objects(model: YOLO, turtle: Turtlebot, R_x: np.ndarray) -> list[ObjectData]:
    """
    Processes images from a Turtlebot's camera and Returns array of detected Objects with their coords.
    """
    img_rgb = turtle.get_rgb_image()
    point_cloud = turtle.get_point_cloud()
    if img_rgb is None or point_cloud is None:
        return []
    results = model(img_rgb)
    detected_objects = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = box.tolist()
            if confidence < 0.5:
                continue
            label = result.names[int(cls)]
            lower_left = int(x1), int(y2)
            height = int(y2 - y1)
            width = int(x2 - x1)
            coords = get_coords(point_cloud, x1, y1, x2, y2, R_x)
            detected_objects.append(
                ObjectData(label, lower_left, width, height, coords)
            )

    # img_cam = generate_anotation(results, detected_objects)
    # cv2.imshow(WINDOW, img_cam)
    # cv2.waitKey(1)
    return detected_objects


if __name__ == "__main__":
    model = load_yolo_model()
    R_x = get_correction_matrix(CAMERA_ANGLE)
    cv2.namedWindow(WINDOW)
    turtle = Turtlebot(rgb=True, depth=False, pc=True)
    while not turtle.is_shutting_down():
        detected_objects = detect_objects(model, turtle, R_x)
        for detected_object in detected_objects:
            print(detected_object)
        print("-" * 20)
