import numpy as np
import cv2 as cv
from cv2 import aruco
import glob
import yaml


aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
board = aruco.GridBoard_create(5, 7, 0.04, 0.01, aruco_dict)
arucoParams = aruco.DetectorParameters_create()

corners_list = []
id_list = []
counter = []
first = True

images = glob.glob('calib_image_*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)

    if first:
        corners_list = corners
        print(type(corners))
        id_list = ids
        first = False
    else:
        corners_list = np.vstack((corners_list, corners))
        id_list = np.vstack((id_list, ids))
    counter.append(len(ids))


counter = np.array(counter)
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, gray.shape, None, None)

data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)


