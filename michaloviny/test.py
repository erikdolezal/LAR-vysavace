from __future__ import print_function

from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

WINDOW = "image"
MODEL_PATH = "best_ones/v11n_120.pt"
DISTANCE_PERCENTILE = 80
HALF_COORD_BOX = 2
X_MIN, X_MAX = -5, 5
Y_MIN, Y_MAX = -1, 5


class ObjectData:
    def __init__(
        self,
        label: str,
        lower_left: tuple[int, int],
        width: int,
        height: int,
        distance: float,
        coords: np.ndarray,
    ) -> None:
        self.label = label
        self.lower_left = lower_left
        self.width = width
        self.height = height
        self.distance = distance
        self.coords = coords

    def __repr__(self) -> str:
        return f"{self.label}: Pos {self.coords}m, Distance {self.distance:.2f}m"


def generate_top_down_view(objects):
    """
    Generates a top-down visualization of objects and returns an image.
    """
    img_size = 640
    margin = 50
    scale_x = (img_size - 2 * margin) / (X_MAX - X_MIN)
    scale_y = (img_size - 2 * margin) / (Y_MAX - Y_MIN)

    # Create blank image (white background)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    def convert_coords(x, y):
        img_x = int((x - X_MIN) * scale_x) + margin
        img_y = img_size - (int((y - Y_MIN) * scale_y) + margin)
        return img_x, img_y

    for x, y, obj_type in objects:
        img_x, img_y = convert_coords(x, y)
        if obj_type == "blue":
            color = (255, 0, 0)
        elif obj_type == "green":
            color = (0, 255, 0)
        elif obj_type == "red":
            color = (0, 0, 255)
        elif obj_type == "ball_y":
            color = (44, 208, 250)
        elif obj_type == "ball_r":
            color = (44, 85, 250)
        else:
            print(f"Unknown object type: {obj_type}")
            color = (135, 135, 135)
        cv2.circle(img, (img_x, img_y), 20, color, -1)
        cv2.putText(
            img,
            f"({x:.1f}, {y:.1f})",
            (img_x + 25, img_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    robot_x, robot_y = convert_coords(0, 0)
    cv2.circle(img, (robot_x, robot_y), 25, (0, 0, 0), -1)
    cv2.putText(
        img,
        "(0.00, 0.00)",
        (robot_x + 25, robot_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return img


def generate_anotation(
    img: np.ndarray, detected_objects: list[ObjectData]
) -> np.ndarray:
    """
    Generates anotations on image.
    """
    for detected_object in detected_objects:
        x, y = detected_object.lower_left
        cv2.putText(
            img,
            f"c:{x:.1f}, {y:.1f}m; {detected_object.distance:.1f}m",
            (int(x), int(y - detected_object.height) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )
    return img


def show(img_cam, space):
    """
    Shows the camera image and top-down view.
    """
    img = np.concatenate((img_cam, space), axis=1)

    h1, w1, c1 = img_cam.shape
    h2, w2, c2 = space.shape
    h, w = h1 + h2, max(w1, w2)
    out_image = np.zeros((h, w, c1))
    out_image[
        :h1,
        :w1,
    ] = img_cam
    out_image[
        h1 : h1 + h2,
        :w2,
    ] = space

    cv2.imshow(WINDOW, img)
    cv2.waitKey(1)


def get_object_distance(
    depth_image: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> float:
    """
    Calculate the distance to an object within a specified region of a depth image.
    """
    object_depth = depth_image[y1:y2, x1:x2]
    valid_depths = object_depth[object_depth > 0]
    if valid_depths.size > 0:
        return float(np.percentile(valid_depths, DISTANCE_PERCENTILE))
    return -1


def get_coords(
    point_cloud: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """
    Get the coordinates of the object within a specified region of a point cloud.
    """
    y_middle = (y1 + y2) // 2
    x_middle = (x1 + x2) // 2
    region = point_cloud[
        y_middle - HALF_COORD_BOX : y_middle + HALF_COORD_BOX,
        x_middle - HALF_COORD_BOX : x_middle + HALF_COORD_BOX,
        :,
    ]
    median_coords = np.median(region, axis=(0, 1))
    return median_coords


def load_yolo_model(model_path=MODEL_PATH) -> YOLO:
    return YOLO(model_path)


def detect_objects(model: YOLO, turtle: Turtlebot) -> list[ObjectData]:
    """
    Processes images from a Turtlebot's camera and Returns array of detected Objects with their coords.
    """
    img_rgb = turtle.get_rgb_image()
    img_depth = turtle.get_depth_image()
    point_cloud = turtle.get_point_cloud()
    if img_rgb is None or img_depth is None or point_cloud is None:
        return []
    results = model(img_rgb)
    detected_objects = []
    for result in results:
        img_cam = result.plot()
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = box.tolist()
            label = result.names[int(cls)]
            lower_left = int(x1), int(y2)
            height = int(y2 - y1)
            width = int(x2 - x1)
            distance = get_object_distance(img_depth, x1, y1, x2, y2)
            coords = get_coords(point_cloud, x1, y1, x2, y2)
            detected_objects.append(
                ObjectData(label, lower_left, width, height, distance, coords)
            )

    img_cam = generate_anotation(img_cam, detected_objects)
    space = generate_top_down_view(detected_objects)
    show(img_cam, space)
    return detected_objects


if __name__ == "__main__":
    model = load_yolo_model()
    cv2.namedWindow(WINDOW)
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    while not turtle.is_shutting_down():
        detected_objects = detect_objects(model, turtle)
        for detected_object in detected_objects:
            print(detected_object)
        print("-" * 20)
