import cv2
import numpy as np 

vid = cv2.VideoCapture(0)

while True:
    _ , img = vid.read()

    cv2.imshow("Picture",img)
    cv2.waitKey(5)
    cv2.destroyAllWindows()