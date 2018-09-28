import cv2
import glob
import numpy as np
import math
import datetime

from multiprocessing import Process, Queue
from os import walk

from haarAlgorithm import faceDetection
from excelAccess import writeMatToExcell, readMatFromExcell

#a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
class Point:
	cl = -1
	dist = -1
	
	def __init__(self,cl,dist):
		self.cl = cl
		self.dist = dist

def histogramNormalization(histogram):
	sum = 0
	for i in range(0,len(histogram)):
		sum += histogram[i] * histogram[i]
	sum = math.sqrt(sum)
	for i in range(0,len(histogram)):
		histogram[i] = histogram[i] / sum
	return histogram
	
def histogramOfGradients(q,image):
    #start = datetime.datetime.now()
	im = np.float32(image) / 255.0
	# Calculate gradient 
	gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
	# Python Calculate gradient magnitude and direction ( in degrees ) 
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	gradHist = [0]*9
	numrows = len(mag)
	numcols = len(mag[0])
	
	for i in range(0,numrows):
		for j in range(0,numcols):
			ang = angle[i,j][0]
			m = mag[i,j][0]	
			
			angl = int(round(ang))
			gradHist[a[angl]] += m
			
			
			#angll = math.floor(angl)
			#nextIndex = angll+1
			#if(angll == 8):
			#	nextIndex = 0
			
			#val = m * (angl - angll)
			#gradHist[nextIndex] += val	
			#gradHist[angll] += (m - val)
			
			
			'''
			x = m/20.0
			
			if(ang > 180):
				ang = 360 - ang
			
			if (ang >= 0 and ang < 20):
				gradHist[0] += x * (20 - ang);
				gradHist[1]  += x * ang;
				
			elif (ang >= 20 and ang < 40):
				gradHist[1]  += x * (40 - ang);
				gradHist[2]  += x * (ang - 20);
				
			elif (ang >= 40 and ang < 60):
				gradHist[2]  += x * (60 - ang);
				gradHist[3]  += x * (ang - 40);
				
			elif (ang >= 60 and ang < 80):
				gradHist[3]  += x * (80 - ang);
				gradHist[4]  += x * (ang - 60);

			elif (ang >= 80 and ang < 100):
				gradHist[4] += x * (100 - ang);
				gradHist[5] += x * (ang - 80);

			elif (ang >= 100 and ang < 120):
				gradHist[5] += x * (120 - ang);
				gradHist[6] += x * (ang - 100);

			elif (ang >= 120 and ang < 140):
				gradHist[6] += x * (140 - ang);
				gradHist[7] += x * (ang - 120);

			elif (ang >= 140 and ang < 160):
				gradHist[7] += x * (160 - ang);
				gradHist[8] += x * (ang - 140);

			elif (ang >= 160 and ang <= 180):
				gradHist[8] += x * (180 - ang);
				gradHist[0] += x * (ang - 160);
			'''

	gradHist = histogramNormalization(gradHist)
	q.put(gradHist)
	#end = datetime.datetime.now()
	#print("HOOOOG:")
        #print(end - start)
	#return gradHist;


def histogramOfGradients1(image):
	im = np.float32(image) / 255.0
	
	# Calculate gradient 
	gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
	# Python Calculate gradient magnitude and direction ( in degrees ) 
	mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

	gradHist = [0]*36
	numrows = len(mag)
	numcols = len(mag[0])
	print(angle)
	h1 = int(numcols / 2)
	h2 = h1 - 1
	r1 = int(numrows / 2)
	r2 = r1 - 1
	
	for i in range(0,numrows):
		for j in range(0,numcols):
			ang = angle[i,j][0]
			m = mag[i,j][0]
			#x = m / 20.0
			#x1 = int(x)
			#print(x)
			#gradHist[0] += x * (20 - ang)
			if(ang <= 180):
				angl = ang / 20.0
				angll = int(angl)
				
				gradHist[angll] = m * (angl - angll)
				nextIndex = angll+1
				if(angll == 8):
					nextIndex = 0
					
				gradHist[nextIndex] = m - gradHist[angll]
			'''
			if (ang >= 0 and ang < 20):
				gradHist[0] += x * (20 - ang);
				gradHist[1]  += x * ang;
				
			elif (ang >= 20 and ang < 40):
				gradHist[1]  += x * (40 - ang);
				gradHist[2]  += x * (ang - 20);
				
			elif (ang >= 40 and ang < 60):
				gradHist[2]  += x * (60 - ang);
				gradHist[3]  += x * (ang - 40);
				
			elif (ang >= 60 and ang < 80):
				gradHist[3]  += x * (80 - ang);
				gradHist[4]  += x * (ang - 60);

			elif (ang >= 80 and ang < 100):
				gradHist[4] += x * (100 - ang);
				gradHist[5] += x * (ang - 80);

			elif (ang >= 100 and ang < 120):
				gradHist[5] += x * (120 - ang);
				gradHist[6] += x * (ang - 100);

			elif (ang >= 120 and ang < 140):
				gradHist[6] += x * (140 - ang);
				gradHist[7] += x * (ang - 120);

			elif (ang >= 140 and ang < 160):
				gradHist[7] += x * (160 - ang);
				gradHist[8] += x * (ang - 140);

			elif (ang >= 160 and ang <= 180):
				gradHist[8] += x * (180 - ang);
				gradHist[0] += x * (ang - 160);
			'''
	
			
	gradHist = histogramNormalization(gradHist)
	return gradHist;
	
def cropImage(img,y,h,x,w):
	image = img[y:y+h, x:x+w]
	return image


def calcDistance(x,y):
	sum = 0
	l = len(x)
	for i in range(0,l):
		#sum += (x[i]-y[i]) * (x[i]-y[i])
		sum += (float(x[i])-float(y[i])) * (float(x[i])-float(y[i]))
	return math.sqrt(sum);


def hogBuildFeatureVectors(trainingImageFolderName, storageFeatureVectorsName):
	
	f = []
	for(dirpath, dirnames, filenames) in walk(trainingImageFolderName):
		f.extend(dirnames)
		break
	
	nrPersons = len(f)
	trainingHist = np.zeros((0,37))

	for i in range(0,nrPersons):
		srcFiles = trainingImageFolderName + '/' + f[i] + '/*ppm'
		print(srcFiles)
		files = glob.glob(srcFiles)
		
		for file in files:
			image = cv2.imread(file)
			image = faceDetection(image)
			
			width = len(image[0])
			height = len(image)
			
			leftTop = cropImage(image, 0, int(height / 2 - 1), 0, int(width / 2 -1))
			rightTop = cropImage(image, 0, int(height / 2), int(width / 2), int(width /2 ))
			leftBottom = cropImage(image, int(height / 2), int(height / 2), 0, int(width /2 ))
			rightBottom = cropImage(image, int(height / 2), int(height / 2), int(width /2 ), int(width /2 ))
					
			leftTopHist = histogramOfGradients(leftTop)
			rightTopHist = histogramOfGradients(rightTop)
			leftBottomHist = histogramOfGradients(leftBottom)
			rightBottomHist = histogramOfGradients(rightBottom)
			
			personIndex = [f[i]]
			hist = np.concatenate([personIndex,leftTopHist,rightTopHist,leftBottomHist,rightBottomHist])
			trainingHist = np.vstack([trainingHist,hist])
	
	writeMatToExcell(storageFeatureVectorsName + '.xlsx',trainingHist)

def hogFaceRecognition(image, storageFeatureVector, storageFeatureVectorsName):

	start = datetime.datetime.now()
	
	if(storageFeatureVector == None):
		storageFeatureVector = readMatFromExcell(storageFeatureVectorsName + '.xlsx')
		
	image = faceDetection(image)
	
	width = len(image[0])
	height = len(image)
	
	leftTop = cropImage(image, 0, int(height / 2 - 1), 0, int(width / 2 -1))
	rightTop = cropImage(image, 0, int(height / 2), int(width / 2), int(width /2 ))
	leftBottom = cropImage(image, int(height / 2), int(height / 2), 0, int(width /2 ))
	rightBottom = cropImage(image, int(height / 2), int(height / 2), int(width /2 ), int(width /2 ))

	q1 = Queue()
	q2 = Queue()
	q3 = Queue()
	q4 = Queue()
	p1 = Process(target=histogramOfGradients, args=(q1, leftTop,))
	p2 = Process(target=histogramOfGradients, args=(q2, rightTop,))
	p3 = Process(target=histogramOfGradients, args=(q3, leftTop,))
	p4 = Process(target=histogramOfGradients, args=(q4, rightBottom,))
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

	hist = np.concatenate([leftTopHist,rightTopHist,leftBottomHist,rightBottomHist])
	
	#results = [Point(-1,10),Point(-1,10),Point(-1,10)]
	results = []
	#hist = histogramOfGradients(image)
	l = len(storageFeatureVector)
	for i in range(0, l):
		dist = calcDistance(hist,storageFeatureVector[i][1:37])
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
	print("Hog time:")
	print(end - start)
	
	return results[0:3]

def hogTestImages(testingImageFolderName, storageFeatureVectorsName):
	storageFeatureVector = readMatFromExcell(storageFeatureVectorsName + '.xlsx')
	
	srcFiles = testingImageFolderName + '/*ppm'
	files = glob.glob(srcFiles)
		
	for file in files:
		image = cv2.imread(file)	
		results = hogFaceRecognition(image, storageFeatureVector, None)
		return results;
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
		



	
