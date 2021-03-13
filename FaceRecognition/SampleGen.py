### This Code is used to Generate Sample Data Set which is used to Train our system.

import cv2

face_class=cv2.CascadeClassifier('Classifier\haarcascade_frontalface_default.xml') ## Importing trained dataset of front face

##Global Variables
offset=20 ## adjust the offset to get full face
samples=0
identification=2 ### set different values for different persons

### This Function Detects and Crop the face from the image

def detect_face_and_crop(img):
    
        global identification,samples
        
        face_img=img.copy()
        gray=cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB) ##converting to grayscale
        faces=face_class.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE) ##Detect the face,parameters like minNeighbers, scaleFactor should be adjusted 

        for x,y,w,h in faces:
            crop=gray[y-offset:y+h+offset,x-offset:x+w+offset] ## Cropping the Face from the gray scale image. Adjust the variable offset if full face is not cropped  
            samples=samples+1
            cv2.imwrite("SampleFace/face-"+str(identification) +'.'+ str(samples) + ".jpg",crop) ## Saving the cropped image
            
            cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),2) ##drawing rectangle to the color image(frame)
        return face_img


cam=cv2.VideoCapture(0)

### Smile please :p
while True:
        ret,frame=cam.read()
        frame=detect_face_and_crop(frame)
        cv2.imshow('face',frame)
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        if samples>50:
            break
        
cam.release()    
cv2.destroyAllWindows()    

