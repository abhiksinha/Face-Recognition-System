import cv2

###This is the main code implemented to Recognize Real Time faces.

##Initial Setup and Variables
recognizer=cv2.face.LBPHFaceRecognizer_create() ### LBPH Face Recognizer 
recognizer.read('trainer/trainer.yml')   ####Loading The Pre-trained dataset (Use SampleGen.py and Train.py to train your image)
face_class=cv2.CascadeClassifier('Classifier\haarcascade_frontalface_default.xml')
name=''
cam=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 


### Main Program

while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_class.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for x,y,w,h in faces:
        identification,confidence= recognizer.predict(gray[y:y+h,x:x+w]) ### Lower Confidence means better result
        
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,255,255),2) ## Drawing a rectangle to the detected face 
        print(confidence)
        
        if confidence<50:      ## Checking confidence level. If else statement adjust according to the users(use dictionary for ease)
            if identification==1:
                name='user1'
            elif identification==2:
                name='user2'
        else:
            name='Undetected'
            
        frame = cv2.putText(frame,name,(x,y+h),font,1,(0,255,0)) ###Writing the name of the recognized person
        cv2.imshow('Real Time Face Recognition',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
                
