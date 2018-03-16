'''
Author: Nandadeep Davuluru
Course: CS 510 Intro to Computer Vision
Instructor: Simon Niklaus
Resources Used: https://stackoverflow.com/a/10806190/3076903
TODO: reduce the number of library dependencies - saved for future. 
Parameters: 
'''

from skimage.feature import hog
from skimage import exposure
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import os, cv2, numpy, random, re, tqdm, pandas as pd
# import matplotlib.pyplot as plotter



INPUT_IMAGES_POSITIVE = './data/testInputPos/'
INPUT_IMAGES_NEGATIVES = './data/testInputNeg/'
FILE_REGEX = re.compile(r'.*\.(jpg|png)+')
pd.set_option('expand_frame_repr', False)
fileTypeCheck = lambda name : re.match(FILE_REGEX, name)
classifier = svm.SVC()
classifier = svm.SVC(kernel='linear', C = 1.0)


def computeHog(pathToPos, sample = True):
	hogFeatures, hogImages, labels = list(), list(), list()
	# redundant af. -_-
	if sample == True:
		for image in tqdm.tqdm(os.listdir(pathToPos)):
			if not fileTypeCheck(image) == None:
				# print(str(INPUT_IMAGES) + '/' + str(image))
				numpyImage = cv2.resize(cv2.imread(str(pathToPos) + '/' + str(image), 0), (128, 128))
				fd, hogImage = hog(numpyImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True, feature_vector=True)
				hogRescaled = exposure.rescale_intensity(hogImage, in_range=(0, 10))
				hogImages.append(hogRescaled.ravel().tolist())
				hogFeatures.append(fd)
				labels.append(1)
	if sample == False:
		for image in tqdm.tqdm(os.listdir(pathToPos)):
			if not fileTypeCheck(image) == None:
				numpyImage = cv2.resize(cv2.imread(str(pathToPos) + '/' + str(image), 0), (128, 128))
				fd, hogImage = hog(numpyImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True, feature_vector=True)
				hogRescaled = exposure.rescale_intensity(hogImage, in_range=(0, 10))
				hogImages.append(hogRescaled.ravel().tolist())
				hogFeatures.append(fd)
				labels.append(-1)

	return hogFeatures, hogImages, labels

def generateDataset(hogFeaturesPos, poslabels, hogFeaturesNeg, neglabels):

	dataset = pd.DataFrame(columns = ['feature', 'label'])
	counter = 0 # bypassing pandas indexing 
	for idx, (feature, label) in enumerate(zip(hogFeaturesPos, poslabels)):
		dataset.loc[idx] = feature, label
		counter = idx
	counter = counter + 1
	for idx, (feature, label) in enumerate(zip(hogFeaturesNeg, neglabels)):		
		dataset.loc[idx + counter] = feature, label

	return shuffle(dataset)


hogFeaturesPos, hogImagesPos, poslabels = computeHog(INPUT_IMAGES_POSITIVE, sample = True)
hogFeaturesNeg, hogImagesNeg, neglabels = computeHog(INPUT_IMAGES_NEGATIVES, sample = False)

dataset = generateDataset(hogFeaturesPos, poslabels, hogFeaturesNeg, neglabels)
train, test = train_test_split(dataset, test_size=0.2)
# split train and test here
featuresTrain, labelsTrain = list(), list()
for idx, row in train.iterrows():
	featuresTrain.append(row['feature'].tolist())
	labelsTrain.append(row['label'])

featuresTest, labelsTest = list(), list()
for idx, row in train.iterrows():
	featuresTest.append(row['feature'].tolist())
	labelsTest.append(row['label'])


classifier.fit(featuresTrain, labelsTrain)
pred = classifier.predict(featuresTest)

print("Accuracy: "+str(accuracy_score(labelsTest, pred)))
print('\n')
print(classification_report(labelsTest, pred))
