import cv2

face_cascade = cv2.CascadeClassifier('Libraries/haarcascade_frontalface_default.xml')

# Haar Cascade for Face Detection. Return the croped image
def faceDetection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	crop = []
	for (x,y,w,h) in faces:
		crop = image[y:y+h, x:x+w]
	
	if len(crop) == 0:
		return image
	else:
		return crop