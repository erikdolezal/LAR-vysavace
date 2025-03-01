from __future__ import print_function

from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
from ultralytics import YOLO
import cv2

WINDOW = 'image'
MODEL_PATH = 'best_ones/v11n_120.pt'
DISTANCE_PERCENTILE = 80



class ObjectData:
    def __init__(self, label:str, lower_left:tuple[int, int], width:int, height:int, distance:float):
        self.label = label
        self.lower_left = lower_left
        self.width = width
        self.height = height
        self.distance = distance
        
    def __repr__(self) -> str:
        return f"{self.label}: Pos {self.lower_left}, Distance {self.distance:.2f}m"

def load_yolo_model(model_path=MODEL_PATH)-> YOLO:
    return YOLO(model_path)

def get_object_distance(depth_image: np.ndarray, x1:int, y1:int, x2:int, y2:int) -> float:
    """
    Calculate the distance to an object within a specified region of a depth image.
    """
    object_depth = depth_image[y1:y2, x1:x2]
    valid_depths = object_depth[object_depth > 0]
    if valid_depths.size > 0:
        return float(np.percentile(valid_depths, DISTANCE_PERCENTILE))
    return -1


def label_camera(model: YOLO, turtle: Turtlebot) -> list[ObjectData]:
    """
    Processes images from a Turtlebot's camera using a YOLO model and Returns array of detected Objects with their distance.
    """
    img_rgb = turtle.get_rgb_image()
    img_depth = turtle.get_depth_image()
    if img_rgb is None or img_depth is None:
        return []
    results = model(img_rgb)
    detected_objects = []
    for result in results:
        img = result.plot()
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = box.tolist()
            label = result.names[int(cls)]
            lower_left = int(x1), int(y2)
            height = int(y2 - y1)
            width = int(x2 - x1)
            distance = get_object_distance(img_depth, x1, y1, x2, y2)
            detected_objects.append(ObjectData(label, lower_left, width, height, distance))
            cv2.putText(img, f"{distance:.2f}m", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)
    return detected_objects

if __name__ == '__main__':
    model = load_yolo_model()
    cv2.namedWindow(WINDOW)
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    while not turtle.is_shutting_down():
        detected_objects = label_camera(model, turtle)
        for detected_object in detected_objects:
            print(detected_object)
        print("-"*20)
