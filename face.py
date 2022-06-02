#to run this program you need to install openCV library using "pip install opencv-python" this comment
#and you need to have trained "haarcascade_frontalface_default.xml" this file look at 11th line

import cv2
import numpy as np


#Starting video camera
cap = cv2.VideoCapture(0)




#Input cascade face detection model
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




while(True):

    ret, camera = cap.read()
    camera = cv2.flip(camera, 1)
    frame = cv2.resize(camera, (640, 480)) 


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray)
    

    
    #drawing bounding box for detected face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    #if you press q the program finishes
    cv2.imshow('Face Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

