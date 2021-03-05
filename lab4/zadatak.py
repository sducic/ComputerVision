import numpy as np
import cv2
import time

img = cv2.imread('lab4.png')

crop_img = img[165:885, 90:1530]            #1440x720
# cv2.imshow("cropped", crop_img)
# cv2.imwrite("crop.png", crop_img)


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Ucitavanje
rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
net = cv2.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')

(winW, winH) = (180, 180)           #180x180px
crvena=(0,0,255)
zuta=(0,255,255)
dog="DOG"
cat="CAT"

for resized in pyramid(crop_img, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=180, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # GoogLeNet
        blob = cv2.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))
        net.setInput(blob)
        preds = net.forward()
        index = np.argsort(preds[0])[::-1][0]

        if preds[0][index] >= 0.51:         #ne prepoznaje psa na >90
            text = "{}".format(classes[index])

            if text == "Siamese cat":
                cv2.putText(crop_img, cat,
                           (
                           int(x * crop_img.shape[1] / resized.shape[1]) + 5, int(y * crop_img.shape[0] / resized.shape[0]) + 23),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, crvena, 2)
                cv2.rectangle(crop_img,
                             (int(x * crop_img.shape[1] / resized.shape[1]), int(y * crop_img.shape[0] / resized.shape[0])),
                             (int((x + winW) * crop_img.shape[1] / resized.shape[1]),
                              int((y + winH) * crop_img.shape[1] / resized.shape[1])),
                             crvena, 2)
                cv2.putText(resized, cat, (x + 5, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, crvena, 2)
                cv2.rectangle(resized, (x, y), (x + winW, y + winH), crvena, 2)

            elif text == "Maltese dog":
                cv2.putText(crop_img, dog,
                                (
                                    int(x * crop_img.shape[1] / resized.shape[1]) + 5,
                                    int(y * crop_img.shape[0] / resized.shape[0]) + 23),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, zuta, 2)
                cv2.rectangle(crop_img,
                                  (int(x * crop_img.shape[1] / resized.shape[1]),
                                   int(y * crop_img.shape[0] / resized.shape[0])),
                                  (int((x + winW) * crop_img.shape[1] / resized.shape[1]),
                                   int((y + winH) * crop_img.shape[1] / resized.shape[1])),
                                  zuta, 2)

                cv2.putText(resized, dog, (x + 5, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zuta, 2)
                cv2.rectangle(resized, (x, y), (x + winW, y + winH), zuta, 2)

                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                #cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)

cv2.imshow("Output", crop_img)
cv2.imwrite("output.jpg", crop_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
