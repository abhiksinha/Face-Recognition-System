import cv2,os
import numpy as np
from PIL import Image 
import pickle
import time

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

pt=0
pic = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) 
while pt<50:
    ret, im =pic.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        id_predict, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        print(confidence)
        
        if (confidence<55):
                if(id_predict==1):
                     Detected='Abhishek'
                elif(id_predict==2):
                    Detected='Subhash'
                elif(id_predict==3):
                     Detected='Rakesh'
                elif(id_predict==4):
                     Detected='Vijju'
                elif(id_predict==5):
                     Detected='Abhinav'
                elif(id_predict==6):
                     Detected='Chotu'     
        else:
            Detected='not found'
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Detected)+"--"+str(confidence), (x,y+h),font, 255) 
        cv2.imshow('im',im)
        cv2.waitKey(10)
        pt=pt+1
        
pic.release()        
cv2.destroyAllWindows()
