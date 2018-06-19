# Note this Example is tested only in Python 2.7
# This is a real time system but processing time may depends upon your processor
# This Code Generates Sample Faces 
import cv2
From ids import id
pic = cv2.VideoCapture(0) 
det=cv2.CascadeClassifier('Classifiers/face.xml')
i=0
offset=50
id=id+1
while True:
    ret, im =pic.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cropFace=det.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in cropFaces:
        i=i+1
        crop=gray[y-offset:y+h+offset,x-offset:x+w+offset]
        cv2.imwrite("dataSet/face-"+id +'.'+ str(i) + ".jpg",crop )
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.waitKey(100)
    if i>50:
        pic.release()
        cv2.destroyAllWindows()
        break
