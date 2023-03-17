import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'images'
imList = [f for f in os.listdir(path) if not f.startswith('.')]
arrIm = []
arrName = []

# Load images and their names from the directory 'images'
for im in imList:
    eq = cv2.imread(f'{path}/{im}')
    arrIm.append(eq)
    arrName.append(os.path.splitext(im)[0])
print(arrName)

# Encode the face features of each image in the list
def encoding(arrIm):
    encolist = []
    for img in arrIm:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        enco = face_recognition.face_encodings(img)[0]
        encolist.append(enco)
    return encolist

# Store the list of encoded face features in a variable
encoWrite = encoding(arrIm)


# Define a function to update attendance records
def attandence(name):
    with open('attandence.csv','r+') as a:
        DataList = a.readlines()
        nameList = []
        for ln in DataList:
            entry = ln.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            # If the name is not already in the attendance record, add a new entry with the current time
            now = datetime.now()
            dtstring = now.strftime('%D   %H : %M : %S')
            a.writelines(f'\n{name},{dtstring}')

vid = cv2.VideoCapture(0)

while True:
    correct, cap1 = vid.read()
    cap1 = cv2.flip(cap1,1)
    # Flip the camera feed horizontally for a more intuitive view
    cap = cv2.resize(cap1,(0,0),None,0.25,0.25)
    cap = cv2.cvtColor(cap,cv2.COLOR_BGR2RGB) 
    # Detect face locations and encode features for each face in the frame
    fcLocC = face_recognition.face_locations(cap)
    fcEncC = face_recognition.face_encodings(cap,fcLocC)

    for encoFace, LocFace in zip(fcEncC,fcLocC):
        comp = face_recognition.compare_faces(encoWrite,encoFace)
        Dis = face_recognition.face_distance(encoWrite,encoFace)
        # If a match is found, mark the face with a rectangle and label it with the name# If a match is found, mark the face with a rectangle and label it with the name
        matIndex = np.argmin(Dis)

        if comp[matIndex]:
            find_name = arrName[matIndex].upper()
            y1,x1,y2,x2 = LocFace
            y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(cap1,(x1,y1),(x2,y2),(0,255,100),2)
            cv2.rectangle(cap1,(x1,y2-35),(x2,y2),(0,255,100),cv2.FILLED)
            cv2.putText(cap1,find_name,(x1-100,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            attandence(find_name)

    if cv2.waitKey(5) & 0xFF==ord('q'):
        break
    cv2.imshow('Webcam',cap1)





       