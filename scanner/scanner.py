import cv2
import numpy as np

img = cv2.imread('scaner.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Kenar tespiti
edges = cv2.Canny(gray, 40, 150, apertureSize=3)

# Hough çizgi tespiti 
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Dikdörtgenleri oluştur
rectangles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if length > 50 and (angle > -30 and angle < 30 or angle > 60 and angle < 120 or angle < -150 or angle > 150):
        rectangles.append(line[0])
        
# Dikdörtgenleri çizdir
for rect in rectangles:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Sonucu göster
cv2.imwrite('./cikti.jpeg', img)
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
