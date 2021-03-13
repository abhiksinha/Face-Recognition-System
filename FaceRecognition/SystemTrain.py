import cv2
import os
import numpy as np


face_class=cv2.CascadeClassifier('Classifier\haarcascade_frontalface_default.xml')
folder = 'Sampleface'
print ("adding faces to the system")

## Function to load all the images and corresponding ids
def load_images_from_folder(folder):
    images = []
    identification=[]
     
    for filename in os.listdir(folder): 
        img = cv2.imread(os.path.join(folder,filename)) 
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
     
        ids = int(os.path.split(filename)[1].split(".")[0].replace("Person_Face", ""))  ## Filtering Id value from File Name
        faces=face_class.detectMultiScale(gray)  ## Detecting faces again just to double check
     
     
        for x,y,w,h in faces:
            image=np.array(gray[y:y+h,x:x+w],'uint8') #converting image to numpy array just to make sure no error happens.
            images.append(image)
            identification.append(ids)
            cv2.imshow("Adding faces to traning set...", image)
            cv2.waitKey(10)

            
    return images,identification

images, identification = load_images_from_folder(folder)
print(type(images[0]))
cv2.imshow("test..", images[0])

cv2.waitKey(10)
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.train(images, np.array(identification))  ###Training the module
recognizer.save('trainer/trainer.yml')  ###Saving the trained module.This will use to Recognize the face later on. 
cv2.destroyAllWindows()
print('completed')



