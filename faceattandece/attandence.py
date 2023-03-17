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
    correct, cap = vid.read()
    cap = cv2.resize(cap,(0,0),None,0.25,0.25)
    cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB) 
    fcLocC = face_recognition.face_locations(cap)
    fcEncC = face_recognition.face_encodings(cap,fcLocC)

    for encoFace, LocFace in zip(fcEncC,fcLocC):
        comp = face_recognition.compare_faces(encoWrite,encoFace)
        Dis = face_recognition.face_distance(encoWrite,encoFace)
        matIndex = np.argmin(Dis)

        if comp[matIndex]:
            find_name = arrName[matIndex].upper()
            print(find_name)






       