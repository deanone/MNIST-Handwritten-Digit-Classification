import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


def load_transform_mnist_data():
	# Load MNIST data
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	# Reshape and change the values of the training images
	train_images = train_images.reshape((60000, 28 * 28))
	train_images = train_images.astype('float32') / 255

	# Reshape and change the values of the test images
	test_images = test_images.reshape((10000, 28 * 28))
	test_images = test_images.astype('float32') / 255

	# Categorical encoding of labels
	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	return (train_images, train_labels), (test_images, test_labels)


def build_model():
	model = models.Sequential()
	model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
	model.add(layers.Dense(10, activation='softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	return model


def main(): 

	(train_images, train_labels), (test_images, test_labels) = load_transform_mnist_data()

	model = build_model()

	# Train model
	model.fit(train_images, train_labels, epochs=10, batch_size=64)

	# Save model
	model.save('mnist_model')

	# Evaluate model
	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('test_acc:', test_acc)


if __name__ == '__main__':
	main()