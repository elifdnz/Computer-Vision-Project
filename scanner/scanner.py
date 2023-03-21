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
    imgBigContour = img.copy()
    contour, hiyearchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContour,contour,-1,(0,255,0),10)

    biggest, maxArea = utlis.biggestContour(contour) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(width,height))

        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)