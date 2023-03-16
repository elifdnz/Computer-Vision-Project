import cv2
import face_recognition
import numpy as np

imgKivanc = face_recognition.load_image_file('kivanc.jpeg')
imgKivanc = cv2.resize(imgKivanc,(640,480))
imgKivanc = cv2.cvtColor(imgKivanc,cv2.COLOR_BGR2RGB)

fcLoc = face_recognition.face_locations(imgKivanc)[0]
fcEnc = face_recognition.face_encodings(imgKivanc)[0]

cv2.rectangle(imgKivanc,(fcLoc[3],fcLoc[0]),(fcLoc[1],fcLoc[2]),(0,0,255),3)

imgtest= face_recognition.load_image_file('test.jpeg')
imgtest = cv2.resize(imgtest,(640,480))
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

fcLocT = face_recognition.face_locations(imgtest)[0]
fcEncT = face_recognition.face_encodings(imgtest)[0]

cv2.rectangle(imgtest,(fcLocT[3],fcLocT[0]),(fcLocT[1],fcLocT[2]),(0,0,255),3)
'''
imgBurak= face_recognition.load_image_file('burak.jpeg')
imgBurak = cv2.resize(imgBurak,(640,480))
imgBurak = cv2.cvtColor(imgBurak,cv2.COLOR_BGR2RGB)

fcLocB = face_recognition.face_locations(imgBurak)[0]
fcEncB = face_recognition.face_encodings(imgBurak)[0]

cv2.rectangle(imgBurak,(fcLocB[3],fcLocB[0]),(fcLocB[1],fcLocB[2]),(0,0,255),3)
'''
result = face_recognition.compare_faces([fcEnc],fcEncT)
dist = face_recognition.face_distance([fcEnc],fcEncT)
text = "This is same"
ntext = "This is different person"
complex = cv2.hconcat([imgKivanc, imgtest])
if result:
    cv2.putText(complex,f'{text}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,100),3)
else:
    cv2.putText(complex,f'{ntext}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,100),3)

cv2.imshow('Compare',complex)
#cv2.imshow('Kivanc',imgKivanc)
#cv2.imshow('Kivanc',imgtest)
#cv2.imshow('Burak',imgBurak)
cv2.waitKey(0)