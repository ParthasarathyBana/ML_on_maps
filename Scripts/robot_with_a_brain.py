#!/usr/bin/env python3

import pickle
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import misc

import keras
from keras import backend as K
from keras.constraints import maxnorm
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils import to_categorical



def load_dataset():
	data_directory = os.listdir("Training_dataset/")
	data_directory=data_directory[:]
	dataset=[]
	for each_folder in data_directory:

	    all_folders = os.listdir("Training_dataset/"+each_folder)

	    for folder in all_folders:
	        img = misc.imread("Training_dataset/"+each_folder+"/"+folder)
	        img = misc.imresize(img, (40, 40))
	        dataset.append((img,each_folder))

	training_set = dataset

	features=[]
	label=[]

	for  feature,image_label in training_set:

	    features.append(feature)

	    label.append(data_directory.index(image_label))


	features=np.array(features)
	label=np.array(label)

	data_set=(features,label)

	save_label = open("int_to_word_out.pickle","wb")
	pickle.dump(data_directory, save_label)
	save_label.close()
	return data_set

def training_network():
	# Sets the value of image dimension ordering convention with tensorflow backend - 'tf'
	K.set_image_dim_ordering('tf')
	# Fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	# load data
	(X_train, Y_train) = load_dataset()

	# Normalize the input data from 0-255 to 0-1.0
	X_train = X_train / 255.0
	# one hot encode outputs
	Y_train = np_utils.to_categorical(Y_train)
	num_classes = Y_train.shape[1]
	# Create the model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(40, 40, 4), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	epochs = 10
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	print(model.summary())
	#callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')]
	callbacks=[keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)]
	# Fit the model
	model.fit(X_train, Y_train, epochs=epochs, batch_size=32,shuffle=True,callbacks=callbacks)

	# Final evaluation of the model
	scores = model.evaluate(X_train, Y_train, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	# serialize model to JSONx
	model_json = model.to_json()
	with open("model_face.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model_face.h5")
	print("Saved model to disk")

def testing_the_model():
	classifier_f = open("int_to_word_out.pickle", "rb")
	int_to_word_out = pickle.load(classifier_f)
	classifier_f.close()

	# load json and create model
	json_file = open('model_face.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model_face.h5")
	print("Model is now loaded in the disk")


	img=os.listdir("Testing_Dataset/")[random.randint(0,726)]
	image=np.array(misc.imread("Testing_Dataset/"+img))
	picture = image
	image = misc.imresize(image, (40, 40))
	image=np.array([image])
	image = image.astype('float32')
	image = image / 255.0

	prediction=loaded_model.predict(image)

	print(prediction)

	print(np.max(prediction))

	print(int_to_word_out[np.argmax(prediction)])

	plt.imshow(picture)
	plt.show()

def main():
	load_dataset()
	training_network()
	testing_the_model()

if __name__ == '__main__':
	main()