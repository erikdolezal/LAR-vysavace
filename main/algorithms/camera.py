import numpy as np
import cv2
import onnxruntime as ort
import time
import torch
import torchvision
from configs.alg_config import vision_config


def softmax(x):
    """
    Compute the softmax of a given input array.
    Args:
        x (numpy.ndarray): Input array or vector for which to compute the softmax.
    Returns:
        numpy.ndarray: An array of the same shape as the input, where each
        element represents the softmax probability of the corresponding input element.
    """

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def xywh2xyxy(x):
    """
    Convert bounding box format from (x_center, y_center, width, height) to
    (x_min, y_min, x_max, y_max).
    Args:
        x (numpy.ndarray): A 2D array where each row represents a bounding box
                           in the format [x_center, y_center, width, height].
    Returns:
        numpy.ndarray: A 2D array where each row represents a bounding box
                       in the format [x_min, y_min, x_max, y_max].
    """

    y = np.zeros(x.shape)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x_min, y_min, x_max, y_max) format
    to (x_center, y_center, width, height) format.
    Args:
        x (numpy.ndarray): A 2D array where each row represents
                           a bounding box in the format [x_min, y_min, x_max, y_max].
    Returns:
        numpy.ndarray: A 2D array  where each row represents
                       a bounding box in the format [x_center, y_center, width, height].
    """

    y = np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


class OnnxCamera:
    """
    A class for performing object detection  using an ONNX model and coordinate transformation.
    Attributes:
        cam_K (numpy.ndarray): Inverse of the camera intrinsic matrix.
        depth_K (numpy.ndarray): Inverse of the depth camera intrinsic matrix.
        cam_to_depth (numpy.ndarray): Transformation matrix from camera to depth coordinates.
        verbose (bool): Flag to enable verbose logging.
        conf_thresh (float): Confidence threshold for filtering detection results.
        model (onnxruntime.InferenceSession): ONNX model inference session.
        input_name (str): Name of the input tensor for the ONNX model.
        output_name (str): Name of the output tensor for the ONNX model.
        input_shape (list): Shape of the input tensor for the ONNX model.
        R_y (numpy.ndarray): Rotation matrix for adjusting camera angle.
        class_map (dict): Mapping of class IDs to custom class indices.
    Methods:
        __init__(model_path, cam_K, depth_K, conf_thresh=0.25, verbose=False):
            Initializes the OnnxCamera object with the given parameters.
        detect(image):
            Performs object detection on the input image using the ONNX model.
        get_detections(image, depth_image):
            Detects objects in the input image, calculates their world coordinates, and visualizes results.
    """

    def __init__(self, model_path, cam_K, depth_K, conf_thresh=0.25, verbose=False):
        self.cam_K = np.linalg.inv(cam_K)
        self.depth_K = np.linalg.inv(depth_K)
        self.cam_to_depth = depth_K @ np.linalg.inv(cam_K)
        self.cam_to_depth[0, 2] -= 10
        self.verbose = verbose
        self.conf_thresh = conf_thresh
        self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        if vision_config["show"]:
            cv2.namedWindow("image")
            cv2.namedWindow("depth")
            cv2.namedWindow("position")
        self.class_map = vision_config["class_map"]

    def detect(self, image):
        """
        Perform object detection on the given image.
        Args:
            image (numpy.ndarray): The input image in BGR format.
        Returns:
            numpy.ndarray: A 2D array where each row represents a detected object with the following columns:
                - [0:4]: Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
                - [4]: Confidence score of the detection.
                - [5]: Class label of the detected object.
        """

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
        cls_prob = softmax(pred[0, 4:, :])
        data_cls = np.argmax(cls_prob, axis=0)
        results = np.hstack(
            (
                pred[0, :4, :].T,
                cls_prob[data_cls, np.arange(data_cls.shape[0])].reshape(-1, 1),
                data_cls.reshape(-1, 1),
            )
        )
        results = results[results[:, 4] > self.conf_thresh]
        results[:, :4] = xywh2xyxy(results[:, :4])
        kept = (
            torchvision.ops.boxes.nms(
                torch.from_numpy(results[:, :4]), torch.from_numpy(results[:, 4]), 0.3
            )
            .detach()
            .numpy()
        )
        results = results[kept, :]
        results[:, [0, 2]] = results[:, [0, 2]] * image.shape[1] / self.input_shape[2]
        results[:, [1, 3]] = results[:, [1, 3]] * image.shape[0] / self.input_shape[3]
        postprocess_time = time.perf_counter() - start

        if self.verbose:
            print(
                f"Preprocess time: {preprocess_time * 1000:.1f} ms Inference time: {inference_time * 1000:.1f} ms Postprocess time: {postprocess_time * 1000:.1f} ms"
            )

        return results

    def get_detections(self, image, depth_image):
        """
        Processes an image and its corresponding depth image to detect objects and
        calculate their world coordinates.
        Args:
            image (numpy.ndarray): The BGR image to process.
            depth_image (numpy.ndarray): The depth image corresponding to the RGB image.
        Returns:
            numpy.ndarray: A 2D array where each row represents the world coordinates and class.
        """

        pred = self.detect(image)
        world_coords = np.zeros((pred.shape[0], 3))
        xywh_preds = xyxy2xywh(pred[:, :4])
        distances = np.zeros((pred.shape[0], 1))
        depth_copy = depth_image.copy() / np.max(depth_image)
        position = np.ones((512, 512, 3))
        cv2.circle(position, (256, 512), 200, (0, 0, 0), 1)
        cv2.circle(position, (256, 512), 300, (0, 0, 0), 1)
        cv2.circle(position, (256, 512), 400, (0, 0, 0), 1)
        cv2.circle(position, (256, 512), 500, (0, 0, 0), 1)
        for i in range(pred.shape[0]):
            x1, y1, x2, y2 = pred[i, :4]
            hom_coords = np.array([[x1, y1, 1], [x2, y2, 1]])

            depth_coords = hom_coords @ self.cam_to_depth.T
            median_distance = (
                np.median(
                    depth_image[
                        int(depth_coords[0, 1]): int(depth_coords[1, 1]),
                        int(depth_coords[0, 0]): int(depth_coords[1, 0]),
                    ]
                )
                / 1000
            )
            median_distance += 0.04 * median_distance
            distances[i] = median_distance
            depth_xy = np.average(depth_coords, axis=0)
            world_coords[i] = (depth_xy @ self.depth_K.T)[[2, 0, 1]]
            world_coords[i, 1] = -world_coords[i, 1]
            world_coords[i] *= median_distance / np.linalg.norm(world_coords[i, 0])
            world_coords[i, 2] = self.class_map[pred[i, 5]]
            distances[i] = np.linalg.norm(world_coords[i, :2])
            if vision_config["show"]:
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
                    vision_config["cls_to_col"][world_coords[i, 2]],
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
                cv2.circle(
                    position,
                    (
                        int(-world_coords[i, 1] * 200) + 256,
                        int(-world_coords[i, 0] * 200) + 512,
                    ),
                    5,
                    vision_config["cls_to_col"][world_coords[i, 2]],
                    -1,
                )

        if vision_config["show"]:
            cv2.imshow("depth", depth_copy)
            cv2.imshow("image", image)
            cv2.imshow("position", position)
            cv2.waitKey(1)
        if self.verbose:
            print(
                f"world coords: {np.hstack((world_coords, np.expand_dims(pred[:, 4], axis=1)))}"
            )
            print(f"distances: {distances}")
        return world_coords


if __name__ == "__main__":
    # from ultralytics import YOLO
    from robolab_turtlebot import Turtlebot

    # import os

    # Load a model
    # model = YOLO("yolo11n.pt")  # load an official model
    # folder = "/best_ones/"
    # files = [a for a in os.listdir(folder) if '.pt' in a and 'v11' in a and '160p' in a]
    # print(files)
    # for model_path in files:
    #    model_path = folder + model_path
    #    model = YOLO(model_path)  # load a custom trained model
    #    # Export the model
    #    model.export(format="onnx")
    # exit(0)
    turtle = Turtlebot(rgb=True, depth=True, pc=True)
    print("Turtle init")
    camera = OnnxCamera(
        "michaloviny/best_ones/v11n_v3_300e_240p_w.onnx",
        verbose=True,
        cam_K=turtle.get_rgb_K(),
        depth_K=turtle.get_depth_K(),
        conf_thresh=0.25,
    )
    print("cam init")
    while not turtle.is_shutting_down():
        st = time.perf_counter()
        img = turtle.get_rgb_image()
        print("image grab time ", time.perf_counter() - st)
        depth_img = turtle.get_depth_image()
        camera.get_detections(img, depth_img)
