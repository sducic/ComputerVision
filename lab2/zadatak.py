import numpy as np
import cv2 as cv

MIN_MATCH_COUNT = 10

slika1 = cv.imread('slika1.JPG', 1)
slika2 = cv.imread('slika2.JPG', 1)
slika3 = cv.imread('slika3.JPG', 1)

# NAPOMENA: Zbog problema sa instalacijom odgovarajuce verzije openCV-a, koristio sam BRISK detektor
detector = cv.BRISK_create()

def findH(img1, img2):
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # findHomography
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # cv.imshow("Output", img3)

    return M


def stichImages(img1, img2, M):
    result = cv.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0] + 60))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result


mat = findH(slika3, slika2)
res = stichImages(slika3, slika2, mat)

mat2 = findH(res, slika1)
res2 = stichImages(res, slika1, mat2)

cv.imshow("Panorama", res2)
cv.imwrite("Panorama.jpg", res2)

cv.waitKey(0)
cv.destroyAllWindows()
