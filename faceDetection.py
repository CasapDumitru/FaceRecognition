import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('DataSet/5/img (14).ppm')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
crop = []
for (x,y,w,h) in faces:
	crop = img[y:y+h, x:x+w]
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

if len(crop) == 0:
	print("Noonee")
else:
	cv2.imshow('img',crop)
	cv2.waitKey(0)
	cv2.destroyAllWindows()