import cv2
import numpy as np

class SlamVisuals():
    def __init__(self, res = (1000,1000)):
        self.img = np.zeros((res[1],res[0],3))
        self.res = res
        self.scale = 10
        self.center = np.array([0.,0.])

    def view_path(self, path, color, closed = True):
        path = np.round(np.append(self.res[0]//2 + (path[:,0] - self.center[0]) * self.scale, self.res[1]//2 - (path[:,1] - self.center[1]) * self.scale, axis=0).reshape(2,-1).transpose())
        path = path.reshape((1,-1,2))
        path = path.astype(int)
        self.img = cv2.polylines(self.img, path, not closed, color, 1)

    def view_points(self, points, color):
        for point in points:
            self.img = cv2.circle(self.img, [int(np.round((point[0]-self.center[0])*self.scale+self.res[0]//2)),int(np.round(self.res[1]//2-(point[1]-self.center[1])*self.scale))], 1, color, 2)

    def show_points(self):
        cv2.imshow('SLAM', self.img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
        self.img = np.zeros((self.res[1],self.res[0],3))

    def rescale(self, cones):
        cones = cones[:,:2].copy()
        x_mid = np.average([np.max(cones[:,0]),np.min(cones[:,0])])
        y_mid = np.average([np.max(cones[:,1]),np.min(cones[:,1])])
        self.center = np.array([x_mid, y_mid])
        cones -= self.center
        self.scale = np.min([(self.res[0]//2)/np.max(np.absolute(cones[:,0]) + 5), (self.res[1]//2)/np.max(np.absolute(cones[:,1]) + 5)])