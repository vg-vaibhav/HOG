import numpy as np
import cv2


def inside(r, p):
    rowx, rowy, roww, rowh = r
    px, py, pw, ph = p
    return rowx > px and rowy > py and rowx + roww < px + pw and rowy + rowh < py + ph


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # 
        # shrinking rect.
        p_w, p_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+p_w, y+p_h), (x+w-p_w, y+h-p_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap=cv2.VideoCapture(0)
    while True:
        _,frame=cap.read()
        found,w=hog.detectMultiScale(frame, winStride=(16 ,16), padding=(32,32), scale=1.05)
        draw_detections(frame,found)
        cv2.imshow('feed',frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
