import numpy as np
import cv2 as cv

# Ucitavanje
img = cv.imread('slika.png')
cv.imshow("Input", img)

# MedianBlur
median = cv.medianBlur(img,5)
cv.imwrite('median.png',median)
cv.imshow('median',median)

# BGR u HSV model
hsv = cv.cvtColor(median, cv.COLOR_BGR2HSV)
cv.imwrite('hsv.png',hsv)
#cv.imshow('hsv',imgHSV)
imgHue = hsv[:, :, 0]
cv.imwrite('hue.png',imgHue)
cv.imshow('hue',imgHue)



# Konvertuje sliku u binarnu koristeci opseg vrednosti

mask = cv.inRange(imgHue, 50, 255)
cv.imwrite('inRange.png',mask)
#cv.imshow("inRange", mask)

# Morfoloska operacija open
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
imgOpen = cv.morphologyEx(src=mask, op=cv.MORPH_OPEN, kernel=kernel)
cv.imwrite('open.png',imgOpen)
cv.imshow("Open", imgOpen)

# Nalazi povezane komponente
imgOut = img.copy()
cntCC, imgCC = cv.connectedComponents(imgOpen, connectivity=4)

# Nalazi i crta bounding box za svaku od identifikovanih komponenti, nalazi najvecu komponentu
maxCnt = 0
maxBBox = None
for cc in range(1, cntCC):
    imgCurr = np.where(imgCC == cc, 255, 0).astype(np.uint8)
    x, y, w, h = cv.boundingRect(imgCurr)
    cnt = imgCurr.sum()
    if cnt > maxCnt:
        maxCnt = cnt
        maxBBox = x, y, w, h
    cv.rectangle(imgOut, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)

# Ispisuje broj identifikovanih komponenti i prikazuje izlaz
cv.putText(imgOut, text='Broj grudvica: '+str(cntCC), org=(5, 17), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
           color=(0, 0, 255), thickness=2)

# Okvir za najvecu komponentu
cv.rectangle(imgOut, pt1=(maxBBox[0], maxBBox[1]), pt2=(maxBBox[0]+maxBBox[2], maxBBox[1]+maxBBox[3]), color=(0, 255, 0), thickness=2)

#Output
cv.imwrite('output.png',imgOut)
cv.imshow("Output", imgOut)

cv.waitKey(0)
cv.destroyAllWindows()
