import cv2
import numpy as np


class SlamVisuals:
    """
    SlamVisuals is a class for visualizing SLAM (Simultaneous Localization and Mapping) data. 
    It provides methods to render paths, points, and dynamically rescale the visualization 
    based on the provided data.
    Attributes:
        img (numpy.ndarray): The image canvas for rendering visualizations.
        res (tuple): The resolution of the visualization canvas (width, height).
        scale (float): The scaling factor for converting real-world coordinates to canvas coordinates.
        center (numpy.ndarray): The center point of the visualization in real-world coordinates.
    Methods:
        __init__(res=(1000, 1000)):
            Initializes the SlamVisuals object.
        view_path(path, color, closed=True):
            Renders a path on the visualization canvas.
        view_points(points, color):
            Renders points on the visualization canvas.
        show_points():
            Displays the current visualization in a window and resets the canvas.
        rescale(cones):
            Rescales the visualization based on the provided cone positions.
    """

    def __init__(self, res=(1000, 1000)):
        self.img = np.zeros((res[1], res[0], 3))
        self.res = res
        self.scale = 10
        self.center = np.array([0.0, 0.0])

    def view_path(self, path, color, closed=True):
        """
        Visualizes a path on an image by drawing a polyline.
        Args:
            path (numpy.ndarray): A 2D array representing the coordinates of the path points.
            color (tuple): A tuple representing the color of the line.
            closed (bool, optional): If True, the polyline will be open.
        Modifies:
            self.img (numpy.ndarray): The image on which the polyline is drawn.
        """

        path = np.round(
            np.append(
                self.res[0] // 2 + (path[:, 0] - self.center[0]) * self.scale,
                self.res[1] // 2 - (path[:, 1] - self.center[1]) * self.scale,
                axis=0,
            )
            .reshape(2, -1)
            .transpose()
        )
        path = path.reshape((1, -1, 2))
        path = path.astype(int)
        self.img = cv2.polylines(self.img, path, not closed, color, 1)

    def view_points(self, points, color):
        """
        Draws a set of points on an image.
        Args:
            points (list of tuples): A list of (x, y) coordinates representing the points to be drawn.
            color (tuple): A tuple representing the color of the points.
        """

        for point in points:
            self.img = cv2.circle(
                self.img,
                [
                    int(
                        np.round(
                            (point[0] - self.center[0]) * self.scale + self.res[0] // 2
                        )
                    ),
                    int(
                        np.round(
                            self.res[1] // 2 - (point[1] - self.center[1]) * self.scale
                        )
                    ),
                ],
                1,
                color,
                2,
            )

    def show_points(self):
        """
        Displays the current SLAM visualization image in a window and resets the image.
        """

        cv2.imshow("SLAM", self.img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            pass
        self.img = np.zeros((self.res[1], self.res[0], 3))

    def rescale(self, cones):
        """
        Rescales the given set of cones to fit within a specified resolution.
        This method adjusts the positions of the cones by centering them around
        their midpoint and scales them to fit within the resolution defined by
        `self.res`. The scaling ensures that the cones remain proportionally
        distributed within the available space.
        Args:
            cones (numpy.ndarray): A 2D array of shape where each row represents coordinates of a cone.      
        Modifies:
            self.center (numpy.ndarray): The center point of the cones after rescaling.
            self.scale (float): The scaling factor for converting real-world coordinates to canvas coordinates.  
        """

        cones = cones[:, :2].copy()
        x_mid = np.average([np.max(cones[:, 0]), np.min(cones[:, 0])])
        y_mid = np.average([np.max(cones[:, 1]), np.min(cones[:, 1])])
        self.center = np.array([x_mid, y_mid])
        cones -= self.center
        self.scale = np.min(
            [
                (self.res[0] // 2) / np.max(np.absolute(cones[:, 0]) + 5),
                (self.res[1] // 2) / np.max(np.absolute(cones[:, 1]) + 5),
            ]
        )
