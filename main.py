import cv2
import numpy as np
import re  
import datetime
import time
import picamera

import pyfirmata
from pyfirmata import STRING_DATA, util


from hogAlgorithm import hogBuildFeatureVectors, hogTestImages, hogFaceRecognition
from lbpAlgorithm import lbpBuildFeatureVectors, lbpTestImages, lbpFaceRecognition
from excelAccess import writeMatToExcell, readMatFromExcell
from haarAlgorithm import faceDetection

trainingImageFolderName = 'DataSet'
lbpStorageFeatureVectorsName = 'lbpFeatureMatrix'
hogStorageFeatureVectorsName = 'hogFeatureMatrix'

board = pyfirmata.Arduino('/dev/ttyUSB0')

#hogBuildFeatureVectors(trainingImageFolderName,hogStorageFeatureVectorsName)
#lbpBuildFeatureVectors(trainingImageFolderName,lbpStorageFeatureVectorsName)
#hogTestImages('Testing', 'hogWithPersonsName')


class User:
	name = ''
	voteNr = 0
	
	def __init__(self,name):
		self.name = name
		
	def addVote(self):
		self.voteNr += 1
		
	def __str__(self):
		s = 'Name: ' + self.name + ' ,  Vote = ' + str(self.voteNr)
		return s

def testImage(img,hogStorageFeatureVector,lbpStorageFeatureVector):
	
	hogResults = hogFaceRecognition(img, hogStorageFeatureVector, 'hogFeatureMatrix')
	lbpResults = lbpFaceRecognition(img, lbpStorageFeatureVector, 'lbpFeatureMatrix')

	finalRes = hogResults + lbpResults
	
	print("#####HOG Results")
	for i in range(0, len(finalRes)):
		if(i == 3):
			print("####LBP Results")
		print(finalRes[i].cl)
		print(finalRes[i].dist)
	
	user = None
	expectedUserHog = None
	expectedUserLbp = None
	
	maxDistanceHog = 0.3
	minDistanceHog = 0.2
	maxDistanceLbp = 0.2
	minDistanceLbp = 0.1
		
	if(hogResults[0].cl == lbpResults[0].cl and hogResults[0].dist < maxDistanceHog and lbpResults[0].dist < maxDistanceLbp):
		user =  hogResults[0].cl
	
	if(user == None):
		if(hogResults[0].dist < minDistanceHog): 
			expectedUserHog = hogResults[0].cl
		
		if(lbpResults[0].dist < minDistanceLbp): 
			expectedUserLbp = lbpResults[0].cl
			
		results = []
		
		for i in range(0, len(finalRes)):
			name = finalRes[i].cl
			exist = False
			for j in range(0, len(results)):
				if(name == results[j].name):
					results[j].addVote()
					exist = True
			if(exist == False):
				u = User(name)
				u.addVote()
				results.append(u)
			
		results.sort(key=lambda x: x.voteNr, reverse = True)
		print(results[0])
		print(results[1])
		
					
		if(expectedUserHog != None and expectedUserLbp != None):
			if((results[0].voteNr != results[1].voteNr) and (results[0].name == expectedUserHog or results[0].name == expectedUserLbp)):
				user = results[0].name
		elif((expectedUserHog != None and results[0].name == expectedUserHog) or (results[0].voteNr == results[1].voteNr and results[1].name == expectedUserHog)):
			user = expectedUserHog
		elif((expectedUserLbp != None and results[0].name == expectedUserLbp) or (results[0].voteNr == results[1].voteNr and results[1].name == expectedUserLbp)):
			user = expectedUserLbp
		
		if(user == None and results[0].voteNr >=3):
			user = results[0].name
	
	return user


camera = picamera.PiCamera()



def whileTrue():
    board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Waiting"))
    hogStorageFeatureVector = readMatFromExcell('hogFeatureMatrix' + '.xlsx')
    lbpStorageFeatureVector = readMatFromExcell('lbpFeatureMatrix' + '.xlsx')
    print('START')
    x = 0
    while True:
            board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Waiting"))
            x+=1
            camera.capture('img.jpg')
            time.sleep(0.1)
            img1 = cv2.imread('img.jpg')
            print(len(img1))
            img = faceDetection(img1)
            if(len(img) != len(img1)):
                    board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("Face Detected"))
                    print(len(img))
                    #cv2.imshow('img',img)
                    #cv2.waitKey(3000)
                    print("BUSY")
                    user = testImage(img,hogStorageFeatureVector,lbpStorageFeatureVector)
                    if(user == None):
                        print('The user is not recognized')
                        board.send_sysex(STRING_DATA, util.str_to_two_byte_iter("NOT RECOGNIZED"))
                    else: 
                        #nr = re.findall('\\d+', user)[0]
                        print('FINAL RESULT: ' + user)
                        board.send_sysex(STRING_DATA, util.str_to_two_byte_iter(user))
                    time.sleep(3)
            print("Ready")   
            if x==10:
                break

whileTrue()


