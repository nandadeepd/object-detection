# Running:
# $ python run-custom-cascade.py

# Import OpenCV
import cv2

# Image file
IMAGE_FILE = 'watch.jpg' # Change this to be your image

# Cascade file
CASCADE_FILE = './data/cascade.xml'

# Cascade item name
CASCADE_ITEM = 'Watch'

# Load image file
image = cv2.imread(IMAGE_FILE)
resized_image = cv2.resize(image, (256, 256)) 

# Convert the image to gray
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Load your cascade file
cascade = cv2.CascadeClassifier(CASCADE_FILE)

# Detect cascade items and put rectangles around them
rectangles = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10,
			minSize=(75, 75))

for (i, (x, y, w, h)) in enumerate(rectangles):
	# Surround cascade with rectangle
	cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(resized_image, CASCADE_ITEM + " #{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# Display the cascade to the user
cv2.imshow(CASCADE_ITEM, resized_image)
cv2.waitKey(0)

# For more:
# http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php
# https://stackoverflow.com/questions/30857908/face-detection-using-cascade-classifier-in-opencv-python