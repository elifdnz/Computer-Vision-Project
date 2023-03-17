import cv2
import face_recognition
import numpy as np
import os


path = 'images'
imList = [f for f in os.listdir(path) if not f.startswith('.')]
'''
imgKivanc = face_recognition.load_image_file('images/kivanc.jpeg')
imgKivanc = cv2.resize(imgKivanc,(640,480))
imgKivanc = cv2.cvtColor(imgKivanc,cv2.COLOR_BGR2RGB)

imgtest= face_recognition.load_image_file('images/kivan.jpeg')
imgtest = cv2.resize(imgtest,(640,480))
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)
'''
arrIm = []
arrName = []

for im in imList:
    eq = cv2.imread(f'{path}/{im}')
    arrIm.append(eq)
    arrName.append(os.path.splitext(im)[0])
print(arrName)


def encoding(arrIm):
    encolist = []
    for img in arrIm:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        enco = face_recognition.face_encodings(img)[0]
        encolist.append(enco)
    return encolist

encoWrite = encoding(arrIm)

vid = cv2.VideoCapture(0)

while True:
    correct, cap1 = vid.read()
    cap = cv2.resize(cap1,(0,0),None,0.25,0.25)
    cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB) 
    fcLocC = face_recognition.face_locations(cap)
    fcEncC = face_recognition.face_encodings(cap,fcLocC)

    for encoFace, LocFace in zip(fcEncC,fcLocC):
        comp = face_recognition.compare_faces(encoWrite,encoFace)
        Dis = face_recognition.face_distance(encoWrite,encoFace)
        matIndex = np.argmin(Dis)

        if comp[matIndex]:
            find_name = arrName[matIndex].upper()
            y1,x1,y2,x2 = LocFace
            y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(cap1,(x1,y1),(x2,y2),(0,255,100),2)
            cv2.rectangle(cap1,(x1,y2-35),(x2,y2),(0,255,100),cv2.FILLED)
            cv2.putText(cap1,find_name,(x1-100,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)

    if cv2.waitKey(5) & 0xFF==ord('q'):
        break
    cv2.imshow('Webcam',cap1)





       