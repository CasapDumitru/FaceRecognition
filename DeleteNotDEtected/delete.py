import os
import numpy as np
import cv2
from os import walk
import glob
#os.remove("aaa.ppm")

face_cascade = cv2.CascadeClassifier('Libraries/haarcascade_frontalface_default.xml')

# Haar Cascade for Face Detection. Return the croped image
def faceDetection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	crop = []
	for (x,y,w,h) in faces:
		crop = image[y:y+h, x:x+w]
	
	if len(crop) == 0:
		return False
	else:
		return True
		
		
def deleteImages():
	folderName = 'images'
	f = []
	for(dirpath, dirnames, filenames) in walk(folderName):
		f.extend(dirnames)
		break
	
	nrOfImages = len(f)

	for i in range(0,nrOfImages):
		srcFiles = folderName + '/' + f[i] + '/*ppm'
		print(srcFiles)
		files = glob.glob(srcFiles)
		
		for file in files:
			#print(file)
			#os.remove(file)
			image = cv2.imread(file)
			x = faceDetection(image)
			if x==False:
				os.remove(file)
			#cv2.imshow('img',image)
			#cv2.waitKey(0)
			#image = faceDetection(image)
			
deleteImages()	
#print("File Removed!")