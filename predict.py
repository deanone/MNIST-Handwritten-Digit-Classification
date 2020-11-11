import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import models
import sys
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
	path = 'images/*.jpg'
	model = models.load_model('mnist_model')
	filenames = glob.glob(path)
	for filename in filenames:
		img = cv2.imread(filename, 0)
		img = cv2.resize(img, (28, 28))
		img = img.reshape(1, img.shape[0] * img.shape[1])
		img = img.astype('float32') / 255
		prediction = model.predict(img)
		prediction = np.argmax(prediction, axis=1)[0]
		print(filename.split('/')[1] + ': ' + str(prediction))


if __name__ == '__main__':
	main()
