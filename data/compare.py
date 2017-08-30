import numpy as np
from PIL import Image


def compare(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err = np.sqrt(err)
	err /= float(imageA.shape[0] * imageA.shape[1])

	return err


def compare2(imageA, imageB):
	err = np.absolute(imageA.astype("float") - imageB.astype("float"))
	print "Out of ", imageA.shape[0]*imageA.shape[1], " pixels only ", np.count_nonzero(err > 20), " are at distance bigger than 20"
	return  np.count_nonzero(err > 20)


pathA = "comparedata/1446742258574.jpg"
pathB = "comparedata/1446742258607.jpg"

imageA = np.asarray(Image.open(pathA))
imageB = np.asarray(Image.open(pathB))

print(compare2(imageA, imageB))