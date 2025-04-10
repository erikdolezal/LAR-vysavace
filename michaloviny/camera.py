from __future__ import print_function

from robolab_turtlebot import Turtlebot, sleep, Rate

import numpy as np
from ultralytics import YOLO
import cv2
import time
import onnxruntime as ort
import time
import torch
import torchvision

SHOW = True

CAMERA_ANGLE = 10
WINDOW = "image"
MODEL_PATH = "michaloviny/best_ones/v11n_120e_160p.pt"
HALF_COORD_BOX = 2
R_PILLAR = 0.02
R_BALL = 0.08


label_map = {"green": 0, "red": 1, "blue": 2, "ball_y": 3, "ball_r": 3}
cls_to_col = {0: (0, 255, 0), 1: (0,0,255), 2: (255,0,0), 3:(255,255,255)}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def xywh2xyxy(x):
    # Convert xywh to xyxy format
    y = np.zeros(x.shape)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def xyxy2xywh(x):
    # Convert xyxy to xywh format
    y = np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

class OnnxCamera:
    def __init__(self, model_path, cam_K, depth_K, conf_thresh=0.25, verbose=False):
        self.cam_K = np.linalg.inv(cam_K)
        self.depth_K = np.linalg.inv(depth_K)
        self.cam_to_depth = depth_K @ np.linalg.inv(cam_K)
        self.cam_to_depth[0,2] -= 10
        self.verbose = verbose
        self.conf_thresh = conf_thresh
        self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.deg2rad(CAMERA_ANGLE)), -np.sin(np.deg2rad(CAMERA_ANGLE))],
                [0, np.sin(np.deg2rad(CAMERA_ANGLE)), np.cos(np.deg2rad(CAMERA_ANGLE))],
            ]
        )
        if SHOW:
            cv2.namedWindow(WINDOW)
            cv2.namedWindow("depth")
        self.class_map = {3:0, 2:2, 1:3, 4:1, 0:3}

    def detect(self, image):
        start = time.perf_counter()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(image, (self.input_shape[2], self.input_shape[3]))
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.float32) / 255.0
        preprocess_time = time.perf_counter() - start

        start = time.perf_counter()
        pred = self.model.run([self.output_name], {self.input_name: input_image})[0]
        inference_time = time.perf_counter() - start

        start = time.perf_counter()
        cls_prob = softmax(pred[0, 4:,:])
        data_cls = np.argmax(cls_prob, axis=0)
        results = np.hstack((pred[0, :4, :].T, cls_prob[data_cls, np.arange(data_cls.shape[0])].reshape(-1,1), data_cls.reshape(-1, 1)))
        results = results[results[:, 4] > self.conf_thresh]
        results[:, :4] = xywh2xyxy(results[:, :4])
        kept = torchvision.ops.boxes.nms(torch.from_numpy(results[:, :4]), torch.from_numpy(results[:, 4]), 0.3).detach().numpy()
        results = results[kept, :]
        results[:,[0,2]] = results[:,[0,2]] * image.shape[1] / self.input_shape[2]
        results[:,[1,3]] = results[:,[1,3]] * image.shape[0] / self.input_shape[3]
        postprocess_time = time.perf_counter() - start

        if self.verbose:
            print(f"Preprocess time: {preprocess_time*1000:.1f} ms Inference time: {inference_time*1000:.1f} ms Postprocess time: {postprocess_time*1000:.1f} ms")
        
        return results
    
    def adjust_coords(self, coords):
        if coords[0] == np.nan:
            return coords
        if coords[2] in [0, 1]:
            angle_z = np.arcsin(np.abs(coords[2]) / np.linalg.norm(coords))
            r_ball = R_BALL * np.cos(angle_z)
            coords += np.array([r_ball, r_ball, 0])
        else:
            coords += np.array([R_PILLAR, R_PILLAR, 0])
        return coords

    def get_detections(self, image, depth_image):
        pred = self.detect(image)
        world_coords = np.zeros((pred.shape[0], 3))
        xywh_preds = xyxy2xywh(pred[:, :4])
        distances = np.zeros((pred.shape[0], 1))
        depth_copy = depth_image.copy()/np.max(depth_image)
        for i in range(pred.shape[0]):
            x1, y1, x2, y2 = pred[i, :4]
            hom_coords = np.array([[x1, y1, 1], [x2, y2, 1]])
            #x, y, w, h = xywh_preds[i,:4]
            #hom_coords = np.array([[x - w/4, y - h/8, 1], [x+w/4, y + h/8, 1]])

            depth_coords = hom_coords @ self.cam_to_depth.T
            median_distance = np.median(depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])])/1000
            #median_distance = np.average(depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])], 
            #                            weights = np.exp(-(depth_image[int(depth_coords[0,1]):int(depth_coords[1,1]), int(depth_coords[0,0]):int(depth_coords[1,0])] - median_distance)**2/100))/1000
            distances[i] = median_distance
            world_coords[i] = (self.R_x @ (np.array([*xywh_preds[i, [0,1]], 1]) @ self.cam_K.T))[[0, 2, 1]]
            world_coords[i,:2] *= median_distance / np.linalg.norm(world_coords[i,1])
            world_coords[i, 2] = self.class_map[pred[i, 5]]
            #world_coords[i] = self.adjust_coords(world_coords[i])
            distances[i] = np.linalg.norm(world_coords[i,:2])
            world_coords[i, 0], world_coords[i, 1] = world_coords[i, 1], -world_coords[i, 0]
            if SHOW:
                x, y = x1, y2
                cv2.putText(
                    image,
                    f"({world_coords[i, 0]:.2f}, {world_coords[i, 1]:.2f}) m",
                    (int(x), int(y - xywh_preds[i, 3]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                cv2.rectangle(
                    image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    cls_to_col[world_coords[i, 2]],
                    2,
                )
                cv2.rectangle(
                    depth_copy,
                    (int(depth_coords[0, 0]), int(depth_coords[0, 1])),
                    (int(depth_coords[1, 0]), int(depth_coords[1, 1])),
                    1,
                    2,
                )
                cv2.putText(
                    depth_copy,
                    f"({distances[i][0]:.2f}) m",
                    (int(depth_coords[0, 0]), int(depth_coords[0, 1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                    2,
                )
        if SHOW:
            cv2.imshow("depth", depth_copy)
            cv2.imshow(WINDOW, image)
            cv2.waitKey(1)
        if self.verbose:
            print(f"world coords: {np.hstack((world_coords, np.expand_dims(pred[:,4], axis=1)))}")
            print(f"distances: {distances}")
        return world_coords



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
    #from ultralytics import YOLO
    import os

    # Load a model
    #model = YOLO("yolo11n.pt")  # load an official model
    #folder = "michaloviny/best_ones/"
    #files = [a for a in os.listdir(folder) if '.pt' in a and 'v11' in a and '160p' in a]
    #print(files)
    #for model_path in files:
    #    model_path = folder + model_path
    #    model = YOLO(model_path)  # load a custom trained model
    #    # Export the model
    #    model.export(format="onnx")
    #exit(0)
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    print("Turtle init")
    camera = OnnxCamera("michaloviny/best_ones/v11n_v3_300e_240p_w.onnx", verbose=True, cam_K=turtle.get_rgb_K(), depth_K=turtle.get_depth_K(), conf_thresh=0.25)
    print("cam init")
    while not turtle.is_shutting_down():
        st = time.perf_counter()
        img = turtle.get_rgb_image()
        print("image grab time ", time.perf_counter() - st)
        depth_img = turtle.get_depth_image()
        camera.get_detections(img, depth_img)
    #camera = Camera(turtle)
    #while not turtle.is_shutting_down():
    #    detected_objects = camera.get_np_objects()
    #    for detected_object in detected_objects:
    #        print(detected_object)
    #    print("-" * 20)
