import numpy as np
import dlib
import cv2

faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Pomocne funkcije za iscrtavanje
def crtajKrugove(tacke, img, index1, index2, velicina, boja, tip):
    for i in range(index1, index2):
        cv2.circle(img, (tacke.part(i).x, tacke.part(i).y), velicina, boja, tip)


def crtajLinije(tacke, img, ind1, ind2, col, debljina):
    for i in range(ind1, ind2):
        cv2.line(img, (tacke.part(i - 1).x, tacke.part(i - 1).y), (tacke.part(i).x, tacke.part(i).y), col, debljina)
    if ind1 > 36:
        cv2.line(img, (tacke.part(ind2 - 1).x, tacke.part(ind2 - 1).y),
                 (tacke.part(ind1 - 1).x, tacke.part(ind1 - 1).y), col, debljina)


def crtajKvadrat(tacke, img, index1, index2, boja):
    for i in range(index1, index2):
        cv2.rectangle(img, (tacke.part(i).x - 3, tacke.part(i).y + 3), (tacke.part(i).x + 3, tacke.part(i).y - 3), boja,
                      -1)


def crtajTrougao(tacke, img, index1, index2, boja):
    for i in range(index1, index2):
        pt1 = (tacke.part(i).x - 3, tacke.part(i).y - 3)
        pt2 = (tacke.part(i).x + 3, tacke.part(i).y - 3)
        pt3 = (tacke.part(i).x, tacke.part(i).y + 3)
        triangle_cnt = np.array([pt1, pt2, pt3])
        cv2.drawContours(img, [triangle_cnt], 0, boja, -1)


def crtajPozadinu():
    pt = np.zeros((17, 2), dtype="int")
    for i in range(0, 17):
        pt[i] = (shape.part(i).x, shape.part(i).y)

    cv2.drawContours(frame, [pt], 0, (255, 0, 255), -1)

    pom1 = (shape.part(16).x, shape.part(16).y)
    pom2 = (shape.part(26).x, shape.part(26).y)
    pom3 = (shape.part(45).x, shape.part(45).y)
    # pom4 = (shape.part(i).x, shape.part(i).y)
    poligon = np.array([pom1, pom2, pom3])
    cv2.drawContours(frame, [poligon], 0, (255, 0, 255), -1)

    pom1 = (shape.part(17).x, shape.part(17).y)
    pom2 = (shape.part(36).x, shape.part(36).y)
    pom3 = (shape.part(0).x, shape.part(0).y)
    # pom4 = (shape.part(i).x, shape.part(i).y)
    poligon = np.array([pom1, pom2, pom3])
    cv2.drawContours(frame, [poligon], 0, (255, 0, 255), -1)

    pom1 = (shape.part(42).x, shape.part(42).y)
    pom2 = (shape.part(22).x, shape.part(22).y)
    pom3 = (shape.part(21).x, shape.part(21).y)
    pom4 = (shape.part(39).x, shape.part(39).y)
    poligon = np.array([pom1, pom2, pom3, pom4])
    cv2.drawContours(frame, [poligon], 0, (255, 0, 255), -1)

    # zuto oko
    pom1 = (shape.part(17).x, shape.part(17).y)
    pom2 = (shape.part(18).x, shape.part(18).y)
    pom3 = (shape.part(19).x, shape.part(19).y)
    pom4 = (shape.part(20).x, shape.part(20).y)
    pom5 = (shape.part(21).x, shape.part(21).y)
    pom6 = (shape.part(39).x, shape.part(39).y)
    pom7 = (shape.part(40).x, shape.part(40).y)
    pom8 = (shape.part(41).x, shape.part(41).y)
    pom9 = (shape.part(36).x, shape.part(36).y)
    poligon = np.array([pom1, pom2, pom3, pom4, pom5, pom6, pom7, pom8, pom9])
    cv2.drawContours(frame, [poligon], 0, (0, 255, 255), -1)

    # zeleno oko
    pom1 = (shape.part(22).x, shape.part(22).y)
    pom2 = (shape.part(23).x, shape.part(23).y)
    pom3 = (shape.part(24).x, shape.part(24).y)
    pom4 = (shape.part(25).x, shape.part(25).y)
    pom5 = (shape.part(26).x, shape.part(26).y)
    pom6 = (shape.part(45).x, shape.part(45).y)
    pom7 = (shape.part(46).x, shape.part(46).y)
    pom8 = (shape.part(47).x, shape.part(47).y)
    pom9 = (shape.part(42).x, shape.part(42).y)
    poligon = np.array([pom1, pom2, pom3, pom4, pom5, pom6, pom7, pom8, pom9])
    cv2.drawContours(frame, [poligon], 0, (0, 255, 0), -1)

    pom1 = (shape.part(27).x, shape.part(27).y)
    pom2 = (shape.part(42).x, shape.part(42).y)
    pom3 = (shape.part(47).x, shape.part(47).y)
    pom4 = (shape.part(46).x, shape.part(46).y)
    pom5 = (shape.part(45).x, shape.part(45).y)
    pom6 = (shape.part(16).x, shape.part(16).y)
    pom7 = (shape.part(30).x, shape.part(30).y)
    poligon = np.array([pom1, pom2, pom3, pom4, pom5, pom6, pom7])
    cv2.drawContours(frame, [poligon], 0, (255, 0, 255), -1)

    pom1 = (shape.part(0).x, shape.part(0).y)
    pom2 = (shape.part(36).x, shape.part(36).y)
    pom3 = (shape.part(41).x, shape.part(41).y)
    pom4 = (shape.part(40).x, shape.part(40).y)
    pom5 = (shape.part(39).x, shape.part(39).y)
    pom6 = (shape.part(27).x, shape.part(27).y)
    pom7 = (shape.part(30).x, shape.part(30).y)
    poligon = np.array([pom1, pom2, pom3, pom4, pom5, pom6, pom7])
    cv2.drawContours(frame, [poligon], 0, (255, 0, 255), -1)

    pom1 = (shape.part(48).x, shape.part(48).y)
    pom2 = (shape.part(59).x, shape.part(59).y)
    pom3 = (shape.part(58).x, shape.part(58).y)
    pom4 = (shape.part(57).x, shape.part(57).y)
    pom5 = (shape.part(56).x, shape.part(56).y)
    pom6 = (shape.part(55).x, shape.part(55).y)
    pom7 = (shape.part(54).x, shape.part(54).y)
    pom8 = (shape.part(64).x, shape.part(64).y)
    pom9 = (shape.part(65).x, shape.part(65).y)
    pom10 = (shape.part(66).x, shape.part(66).y)
    pom11 = (shape.part(67).x, shape.part(67).y)
    pom12 = (shape.part(60).x, shape.part(60).y)
    poligon = np.array([pom1, pom2, pom3, pom4, pom5, pom6, pom7, pom8, pom9, pom10, pom11, pom12])
    cv2.drawContours(frame, [poligon], 0, (0, 255, 0), -1)

    pom1 = (shape.part(42).x, shape.part(42).y)
    pom2 = (shape.part(43).x, shape.part(43).y)
    pom3 = (shape.part(44).x, shape.part(44).y)
    pom4 = (shape.part(45).x, shape.part(45).y)
    pom5 = (shape.part(46).x, shape.part(46).y)
    pom6 = (shape.part(47).x, shape.part(47).y)
    poligon = np.array([pom1, pom2, pom3, pom4, pom5, pom6])
    cv2.drawContours(frame, [poligon], 0, (255, 0, 0), -1)


def crtajOkvir():
    crtajLinije(shape, frame, 1, 17, (255, 0, 0), 2)
    crtajLinije(shape, frame, 28, 31, (0, 255, 255), 2)
    crtajLinije(shape, frame, 31, 36, (0, 255, 0), 2)
    cv2.line(frame, (shape.part(30).x, shape.part(30).y), (shape.part(35).x, shape.part(35).y), (0, 255, 0), 2)
    cv2.line(frame, (shape.part(0).x, shape.part(0).y), (shape.part(17).x, shape.part(17).y), (255, 0, 0), 2)
    cv2.line(frame, (shape.part(21).x, shape.part(21).y), (shape.part(22).x, shape.part(22).y), (255, 0, 0), 2)
    cv2.line(frame, (shape.part(16).x, shape.part(16).y), (shape.part(26).x, shape.part(26).y), (255, 0, 0), 2)
    crtajLinije(shape, frame, 18, 22, (255, 255, 0), 2)
    crtajLinije(shape, frame, 22, 27, (0, 0, 255), 2)
    cv2.line(frame, (shape.part(17).x, shape.part(17).y), (shape.part(36).x, shape.part(36).y), (255, 255, 0), 2)
    cv2.line(frame, (shape.part(21).x, shape.part(21).y), (shape.part(39).x, shape.part(39).y), (255, 255, 0), 2)
    cv2.line(frame, (shape.part(22).x, shape.part(22).y), (shape.part(42).x, shape.part(42).y), (0, 0, 255), 2)
    cv2.line(frame, (shape.part(26).x, shape.part(26).y), (shape.part(45).x, shape.part(45).y), (0, 0, 255), 2)
    crtajLinije(shape, frame, 49, 55, (0, 255, 0), 2)
    crtajLinije(shape, frame, 61, 65, (0, 255, 0), 2)
    cv2.line(frame, (shape.part(42).x, shape.part(42).y), (shape.part(43).x, shape.part(43).y), (255, 0, 255), 2)
    cv2.line(frame, (shape.part(43).x, shape.part(43).y), (shape.part(44).x, shape.part(44).y), (255, 0, 255), 2)
    cv2.line(frame, (shape.part(44).x, shape.part(44).y), (shape.part(45).x, shape.part(45).y), (255, 0, 255), 2)
    cv2.line(frame, (shape.part(21).x, shape.part(21).y), (shape.part(22).x, shape.part(22).y), (255, 0, 0), 2)


cap = cv2.VideoCapture(0)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 67):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = dlib.rectangles()  # konverzija
    for (x, y, w, h) in faces:
        rects.append(dlib.rectangle(x, y, x + w, y + h))

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        # shape = shape_to_np(shape)

        # Funckije za iscrtavanje
        crtajPozadinu()

        # Linije okvira
        crtajOkvir()

        ### kruzici
        crtajKrugove(shape, frame, 48, 55, 3, (170, 170, 170), -1)
        crtajKrugove(shape, frame, 60, 65, 3, (170, 170, 170), -1)
        cv2.circle(frame, (shape.part(46).x, shape.part(46).y), 3, (170, 170, 170), -1)
        cv2.circle(frame, (shape.part(47).x, shape.part(47).y), 3, (170, 170, 170), -1)

        ### kvadrati
        crtajKvadrat(shape, frame, 0, 27, (170, 170, 170))
        crtajKvadrat(shape, frame, 40, 42, (170, 170, 170))
        crtajKvadrat(shape, frame, 27, 36, (170, 170, 170))

        ### trouglovi
        crtajTrougao(shape, frame, 55, 60, (170, 170, 170))
        crtajTrougao(shape, frame, 65, 68, (170, 170, 170))
        crtajTrougao(shape, frame, 42, 46, (170, 170, 170))
        crtajTrougao(shape, frame, 36, 40, (170, 170, 170))

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
