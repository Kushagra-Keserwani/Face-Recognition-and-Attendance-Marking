import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='Image Database'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('AttendanceSheet.csv', 'r+') as sheet:
        myDataList = sheet.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            sheet.writelines(f'\n{name},{dtString}')



EncodedImagesData = findEncodings(images)
print("Images Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    sucess, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrames = face_recognition.face_encodings(imgS,facesCurrFrame)

    for encodeFace, FaceLoc in zip(encodeCurrFrames, facesCurrFrame):
        matches = face_recognition.compare_faces(EncodedImagesData, encodeFace)
        faceDis = face_recognition.face_distance(EncodedImagesData, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=FaceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
            markAttendance(name)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
