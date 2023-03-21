import cv2
import numpy as np
import utlis

scn = cv2.imread('scaner.jpeg')
cap = cv2.VideoCapture(0)
web = True
cap.set(10,160)
height = 640
width = 480

count = 0

utlis.initializeTrackbars()
while True:
    imgB = np.zeros((height,width,3),np.uint8)
    if web :succes,img = cap.read()
    else: img = cap.read(scn)
    img = cv2.resize(img,(width,height))
    imgG = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
    imgBlu = cv2.GaussianBlur(imgG,(5,5),1)
    thrs=utlis.valTrackbars()
    imgThreshold = cv2.Canny(imgBlu,thrs[0],thrs[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgThreshold,kernel,iterations=2)
    imgThreshold = cv2.erode(imgDial,kernel,iterations=1)

    imgContour = img.copy()
    imgBigContoru = img.copy()
    contour, hiyearchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour,contour,-1,(0,255,0),10)

