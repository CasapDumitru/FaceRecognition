import cv2
import glob
import math
import numpy as np
import datetime

from os import walk

from haarAlgorithm import faceDetection
from excelAccess import writeMatToExcell, readMatFromExcell

from multiprocessing import Process, Queue

indexVector = [0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10, 11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15, 16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21, 22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28, 29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35, 36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41, 42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46, 47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57]

cosVec = [1, 1, 0, -1, -1, -1, -0, 1]
sinVec = [0, 1, 1, 1, 0, -1, -1, -1]

class Point:
	cl = -1
	dist = -1
	
	def __init__(self,cl,dist):
		self.cl = cl
		self.dist = dist


def cropImage(img,y,h,x,w):
	image = img[y:y+h, x:x+w]
	return image
	

def bit(b,i):
	return ((b & (1 << i)) != 0)
	
def histogramNormalization(histogram):
	sum = 0
	l = len(histogram)
	for i in range(0,l):
		sum += histogram[i] * histogram[i]
	sum = math.sqrt(sum)
	for i in range(0,l):
		histogram[i] = histogram[i] / sum
		
	return histogram
		

def calcDistance(x,y):
	sum = 0
	l = len(x)
	for i in range(1,l):
		sum += (float(x[i])- float(y[i])) * (float(x[i])-float(y[i])) / (float(x[i]) + float(y[i]))
	return sum;
		

def lbpVectorWithIndexForHist(bitsNr, histSize):
	
	vectorSize = pow(2,8)
	indexVector = [0] * vectorSize
	indexVector[0] = 0
	indexVector[vectorSize-1] = histSize - 2
	count = 1
	
	for i in range(1,vectorSize-1):
		trans = 0
		for j in range(0,bitsNr - 1):
			trans += (bit(i,j) != bit(i,j+1))
		trans += (bit(i,bitsNr - 1) != bit(i,0))
		
		if(trans <=2 ):
			indexVector[i] = count
			count += 1
		else:
			indexVector[i] = histSize - 1
	print(indexVector)
	return indexVector
	
def constructCosVectorPR(P,R):
	vec = [0] * P
	for q in range(0,P):
		vec[q] = R*round(math.cos(2*math.pi*q/P))
	print(vec)
	return vec
	
def constructSinVectorPR(P,R):
	vec = [0] * P
	for q in range(0,P):
		vec[q] = R*round(math.sin(2*math.pi*q/P))
	print(vec)
	return vec
		
def lbpHistogram(queue, image, R, P, indexVector,cosVec,sinVec):
    #start = datetime.datetime.now()
	image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )
	#R = 1
	#P = 8
	histSize = P * (P - 1) + 3
	hist = [1] * histSize
	
	
	numrows = len(image) - R
	numcols = len(image[0]) - R
	
	#indexVector = lbpVectorWithIndexForHist(P, histSize)
	
	
	for i in range(R,numrows):
		for j in range(R,numcols):
			histVal = 0
			currentVal = image[i,j]		
			for q in range(0,P):
				x = j + cosVec[q]
				y = i + sinVec[q]
				
				if (image[y,x] > currentVal):
						histVal += pow(2,q)
				
			hist[indexVector[histVal]] += 1
			
	hist = histogramNormalization(hist)

	#end = datetime.datetime.now()
        #print("LBPPP:")
        #print(end - start)
	queue.put(hist)
	#return hist


def lbpBuildFeatureVectors(trainingImageFolderName, storageFeatureVectorsName):

	f = []
	for(dirpath, dirnames, filenames) in walk(trainingImageFolderName):
		f.extend(dirnames)
		break
	
	nrPersons = len(f)
	
	trainingHist = np.zeros((0,237))

	for i in range(0,nrPersons):
		srcFiles = trainingImageFolderName + '/' + f[i] + '/*jpg'
		print(srcFiles)
		files = glob.glob(srcFiles)
		
		for file in files:
			image = cv2.imread(file)
			image = faceDetection(image)
			
			width = len(image[0])
			height = len(image)
			
			leftTop = cropImage(image, 0, int(height / 2), 0, int(width / 2))
			rightTop = cropImage(image, 0, int(height / 2), int(width / 2 - 1), int(width / 2 ))
			leftBottom = cropImage(image, int(height / 2), int(height / 2), 0, int(width / 2 + 1))
			rightBottom = cropImage(image, int(height / 2 - 1), int(height / 2 - 1), int(width /2 ), int(width /2 ))
					
			R = 1
			P = 8
			histSize = P * (P - 1) + 3
			hist = [0] * histSize
			indexVector = lbpVectorWithIndexForHist(P, histSize)
			
			cosVec = constructCosVectorPR(P,R)
			sinVec = constructSinVectorPR(P,R)

			q1 = Queue()
			q2 = Queue()
			q3 = Queue()
			q4 = Queue()
			p1 = Process(target=lbpHistogram, args=(q1,leftTop,R,P,indexVector,cosVec,sinVec,))
			p2 = Process(target=lbpHistogram, args=(q2,rightTop,R,P,indexVector,cosVec,sinVec,))
			p3 = Process(target=lbpHistogram, args=(q3,leftTop,R,P,indexVector,cosVec,sinVec,))
			p4 = Process(target=lbpHistogram, args=(q4,rightBottom,R,P,indexVector,cosVec,sinVec,))
			p1.start()
			p2.start()
			p3.start()
			p4.start()
			leftTopHist= q1.get()
			rightTopHist = q2.get()
			leftBottomHist = q3.get()
			rightBottomHist = q4.get()
			p1.join()
			p2.join()
			p3.join()
			p4.join()
			'''
			leftTopHist = lbpHistogram(leftTop,R,P,indexVector,cosVec,sinVec)
			rightTopHist = lbpHistogram(rightTop,R,P,indexVector,cosVec,sinVec)
			leftBottomHist = lbpHistogram(leftBottom,R,P,indexVector,cosVec,sinVec)
			rightBottomHist = lbpHistogram(rightBottom,R,P,indexVector,cosVec,sinVec)
			'''
			
			personIndex = [f[i]]
			hist = np.concatenate([personIndex,leftTopHist,rightTopHist,leftBottomHist,rightBottomHist])
			trainingHist = np.vstack([trainingHist,hist])
	
	writeMatToExcell(storageFeatureVectorsName + '.xlsx',trainingHist)

def lbpFaceRecognition(image, storageFeatureVector, storageFeatureVectorsName):
	start = datetime.datetime.now()
	image = faceDetection(image)
	

	if(storageFeatureVector == None):
		storageFeatureVector = readMatFromExcell(storageFeatureVectorsName + '.xlsx')
			
	width = len(image[0])
	height = len(image)
	
	leftTop = cropImage(image, 0, int(height / 2), 0, int(width / 2))
	rightTop = cropImage(image, 0, int(height / 2), int(width / 2 - 1), int(width / 2 ))
	leftBottom = cropImage(image, int(height / 2), int(height / 2), 0, int(width / 2 + 1))
	rightBottom = cropImage(image, int(height / 2 - 1), int(height / 2 - 1), int(width /2 ), int(width /2 ))
	
	R = 1
	P = 8
	'''
	histSize = P * (P - 1) + 3
	hist = [0] * histSize
	indexVector = lbpVectorWithIndexForHist(P, histSize)
	
	cosVec = constructCosVectorPR(P,R)
	sinVec = constructSinVectorPR(P,R)
	'''
	q1 = Queue()
	q2 = Queue()
	q3 = Queue()
	q4 = Queue()
	p1 = Process(target=lbpHistogram, args=(q1,leftTop,R,P,indexVector,cosVec,sinVec,))
	p2 = Process(target=lbpHistogram, args=(q2,rightTop,R,P,indexVector,cosVec,sinVec,))
	p3 = Process(target=lbpHistogram, args=(q3,leftTop,R,P,indexVector,cosVec,sinVec,))
	p4 = Process(target=lbpHistogram, args=(q4,rightBottom,R,P,indexVector,cosVec,sinVec,))
	p1.start()
	p2.start()
	p3.start()
	p4.start()
	leftTopHist= q1.get()
	rightTopHist = q2.get()
	leftBottomHist = q3.get()
	rightBottomHist = q4.get()
	p1.join()
	p2.join()
	p3.join()
	p4.join()
	'''
	leftTopHist = lbpHistogram(leftTop,R,P,indexVector,cosVec,sinVec)
	rightTopHist = lbpHistogram(rightTop,R,P,indexVector,cosVec,sinVec)
	leftBottomHist = lbpHistogram(leftBottom,R,P,indexVector,cosVec,sinVec)
	rightBottomHist = lbpHistogram(rightBottom,R,P,indexVector,cosVec,sinVec)
	'''
	
	
	
	hist = np.concatenate([leftTopHist,rightTopHist,leftBottomHist,rightBottomHist])
	
	results = []
        #results = [Point(-1,10),Point(-1,10),Point(-1,10)]
	
	l = len(storageFeatureVector)
	
	for i in range(0, l):
		dist = calcDistance(hist,storageFeatureVector[i][1:237])
		p = Point(storageFeatureVector[i][0],dist)
		'''
		if(p.dist < results[0].dist):
			results[2] = results[1]
			results[1] = results[0]
			results[0] = p
		elif(p.dist < results[1].dist):
			results[2] = results[1]
			results[1] = p
		elif(p.dist < results[2].dist):
			results[2] = p
		'''	
		
		results.append(p)
		results.sort(key=lambda x: x.dist)
	end = datetime.datetime.now()
	print("LBPPP:")
	print(end - start)
	return results[0:3]
	
def lbpTestImages(testingImageFolderName, storageFeatureVectorsName):
	storageFeatureVector = readMatFromExcell(storageFeatureVectorsName + '.xlsx')
	
	srcFiles = testingImageFolderName + '/*ppm'
	files = glob.glob(srcFiles)
		
	for file in files:
		image = cv2.imread(file)	
		results = lbpFaceRecognition(image, storageFeatureVector, None)
		'''
		print(file)
		print(results[0].cl)
		print(results[0].dist)
		print(results[1].cl)
		print(results[1].dist)
		print(results[2].cl)
		print(results[2].dist)
		print(results[3].cl)
		print(results[3].dist)
		print(results[4].cl)
		print(results[4].dist)
		'''
		return results
