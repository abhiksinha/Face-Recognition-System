### This code is based on Viola Jones Algorithm for Face Detection(not confused with Face Recognition)



import cv2

face_class=cv2.CascadeClassifier('Classifier\haarcascade_frontalface_default.xml') ## importing trained dataset of frontalface

###### Detect face function 

def detect_face(img):
    face_img=img.copy()
    gray=cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB) ##converting to grayscale
    faces=face_class.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE) ##detect the face,parameters like minNeighbers, scaleFactor should be adjusted 
    
    for x,y,w,h in faces:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),2) ##drawing rectangle on the image
    return face_img

###### Code to create a window for real time FaceDetection
cam=cv2.VideoCapture(0) ##initializig camera
while True:
    ret,frame=cam.read() ###reading a frame
    fr=detect_face(frame) ###calling the detect face function for a frame
    cv2.imshow('image',fr)
    if cv2.waitKey(1) & 0xFF==ord('q'):    ###press q to exit
        break
cam.release() 
cv2.destroyAllWindows()
