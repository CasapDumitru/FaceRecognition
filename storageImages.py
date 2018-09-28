import cv2
import numpy as np
import re  
import datetime
import time
#import picamera
import os

from hogAlgorithm import hogBuildFeatureVectors, hogTestImages, hogFaceRecognition
from lbpAlgorithm import lbpBuildFeatureVectors, lbpTestImages, lbpFaceRecognition
from excelAccess import writeMatToExcell, readMatFromExcell
from haarAlgorithm import faceDetection

trainingImageFolderName = 'DataSet'
lbpStorageFeatureVectorsName = 'lbpFeatureMatrix'
hogStorageFeatureVectorsName = 'hogFeatureMatrix'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def whileTrue():
	personName = input('Person Name: ')
	directory = './NewDataSet/' + personName + '/'
	createFolder(directory)
	x = 0
	while True:
		x+=1
		print(x)
		#camera.capture('img.jpg')
		time.sleep(0.1)
		img1 = cv2.imread('TestImage.png')
		print(len(img1))
		img = faceDetection(img1)
		if(len(img) != len(img1)):
			cv2.imwrite(directory + 'img' + str(x) + '.jpg', img1) 
		if x==5:
			break


trainingImageFolderName = 'NewDataSet'

lbpBuildFeatureVectors(trainingImageFolderName,'newLbpTest')

#whileTrue()

