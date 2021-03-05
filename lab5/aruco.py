import numpy as np
import cv2 as cv
from cv2 import aruco
import yaml

tvec = np.array([0, 0, 0], dtype=np.float32)
rvec = np.array([0, 0, 0], dtype=np.float32)

aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
board = aruco.GridBoard_create(5, 7, 0.04, 0.01, aruco_dict)
arucoParams = aruco.DetectorParameters_create()

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

h, w = (1080, 1920)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

videoFile = "Aruco_board.mp4"
cap = cv.VideoCapture(videoFile)

while True:
    ret, frame = cap.read()
    if ret:
        img_aruco = frame
        im_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        dst = cv.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)

        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist,rvec,tvec)
        if ret != 0:
            img_aruco = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
            img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 0.1)

        cv.imshow("Output", img_aruco)

        if cv.waitKey(2) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv.destroyAllWindows()