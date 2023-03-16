import cv2
import face_recognition
import numpy as np
import os


path = 'images'
imList = os.listdir(path)

imgKivanc = face_recognition.load_image_file('images/kivanc.jpeg')
imgKivanc = cv2.resize(imgKivanc,(640,480))
imgKivanc = cv2.cvtColor(imgKivanc,cv2.COLOR_BGR2RGB)

imgtest= face_recognition.load_image_file('images/test.jpeg')
imgtest = cv2.resize(imgtest,(640,480))
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

arrIm = []
arrName = []

for im in imList:
    eq = cv2.imread(f'{path}/{im}')
    arrIm.append(eq)
    arrName.append(os.path.splitext(im)[0])
print(arrName)

