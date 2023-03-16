import cv2
import face_recognition
import numpy as np
imgKivanc = face_recognition.load_image_file('kivanc.jpeg')
imgKivanc = cv2.cvtColor(imgKivanc,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('test.jpeg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

cv2.imshow('Kivanc',imgKivanc)
cv2.imshow('Test',imgtest)
cv2.waitKey(0)