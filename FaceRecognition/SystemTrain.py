import cv2
import os
import numpy as np
from PIL import Image 

recognizer = cv2.createLBPHFaceRecognizer()
cascadePath = "Classifiers/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'Samplefaces'
print "adding you to system"
def get_images_and_labels(path):
# Reading Images and Labels id
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     images = []
     labels = []
     for image_path in image_paths:
     # converting RGB to Gray
         image_pil = Image.open(image_path).convert('L')
      # gray to numpy array for direct Matrix Operation
         image = np.array(image_pil, 'uint8')
          # Seperating Image ids
         lab = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))
         
         
         # Detect the face in the image
         faces = faceCascade.detectMultiScale(image)
         # If face is detected, append the face to images and the label to labels
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(lab)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     return images, labels


images, labels = get_images_and_labels(path)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()
