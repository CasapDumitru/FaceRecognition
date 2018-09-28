import cv2
import numpy as np
import re  
import datetime
import time

from hogAlgorithm import hogBuildFeatureVectors, hogTestImages, hogFaceRecognition
from lbpAlgorithm import lbpBuildFeatureVectors, lbpTestImages, lbpFaceRecognition
from excelAccess import writeMatToExcell, readMatFromExcell
from haarAlgorithm import faceDetection

trainingImageFolderName = 'DataSet'
lbpStorageFeatureVectorsName = 'lbpFeatureMatrix'
hogStorageFeatureVectorsName = 'hogFeatureMatrix'

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

def testImage1(img,hogStorageFeatureVector,lbpStorageFeatureVector):
	
	hogResults = hogFaceRecognition(img, hogStorageFeatureVector, 'hogFeatureMatrix')
	lbpResults = lbpFaceRecognition(img, lbpStorageFeatureVector, 'lbpFeatureMatrix')

	finalRes = hogResults + lbpResults
	
	'''
	print("#####HOG Results")
	for i in range(0, len(finalRes)):
		if(i == 3):
			print("####LBP Results")
		print(finalRes[i].cl)
		print(finalRes[i].dist)
	'''
	
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



start = datetime.datetime.now()
hogStorageFeatureVector = readMatFromExcell('hogFeatureMatrix' + '.xlsx')


end = datetime.datetime.now()
print("HOG Extract Feature Vector:")
print(end - start)
start = datetime.datetime.now()


lbpStorageFeatureVector = readMatFromExcell('lbpFeatureMatrix' + '.xlsx')

end = datetime.datetime.now()
print("LBP Extract Feature Vector:")
print(end - start)
start = datetime.datetime.now()

img1 = cv2.imread('TestImage.png')
print(len(img1))
print(len(img1[0]))

img = faceDetection(img1)

end = datetime.datetime.now()
print("Face Detection:")
print(end - start)
start = datetime.datetime.now()

print(len(img))
print(len(img[0]))

user = 'dsa'
hogResults = hogFaceRecognition(img, hogStorageFeatureVector, 'hogFeatureMatrix')
#lbpResults = lbpFaceRecognition(img, lbpStorageFeatureVector, 'lbpFeatureMatrix')
'''user = None
if __name__ == '__main__':
	user = testImage1(img,hogStorageFeatureVector,lbpStorageFeatureVector)
'''
end = datetime.datetime.now()
print("Face Recognition:")
print(end - start)

if(user == None):
    print('The user is not recognized')
else: 
    print('FINAL RESULT: ' + user)
