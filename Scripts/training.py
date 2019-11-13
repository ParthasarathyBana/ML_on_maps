#! /usr/bin/env python3

#from typing import Any, Union
import numpy as np
import os
import random
import pickle
import math
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy import misc
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
import keras
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, \
    GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils

import tensorflow as tf
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
from collections import defaultdict

def load_robot_centric_dataset():
    data_directory = os.listdir("image_dataset/robot_centric/")
    data_directory = data_directory[:]
    # print(data_directory)
    dataset = defaultdict(list)
    training_dataset = []
    for folder in data_directory:
        all_image_paths = os.listdir("image_dataset/robot_centric/" + folder)
        
        for image in all_image_paths:
            path = "image_dataset/robot_centric/" + folder + "/" + image
            dataset[folder].append(path)

    req_files = min(len(val) for key,val in dataset.items())
    
    min_idx = 0
    # list of random numbers (0, max_files, len=req_files)
    for folder, image_paths in dataset.items():
        max_files = len(image_paths)
        index_list = random.sample(range(0, max_files), req_files)
        for i in index_list:
            img = cv2.imread(dataset[folder][i])
            img = cv2.resize(img, (150, 150))
            training_dataset.append((img, folder))
    features = []
    label = []
    for feature, image_label in training_dataset:
        features.append(feature)
        label.append(data_directory.index(image_label))

    X_train = np.array(features)
    Y_train = np.array(label)


    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, shuffle=True)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    save_label = open("plain_robot_model.pickle", "wb")
    pickle.dump(data_directory, save_label)
    save_label.close()
    return X_test, Y_test, X_train, Y_train

def load_map_centric_dataset():
    data_directory = os.listdir("image_dataset/map_centric/")
    data_directory = data_directory[:]
    # print(data_directory)
    dataset = defaultdict(list)
    training_dataset = []
    for folder in data_directory:
        all_image_paths = os.listdir("image_dataset/map_centric/" + folder)
        
        for image in all_image_paths:
            path = "image_dataset/map_centric/" + folder + "/" + image
            dataset[folder].append(path)

    req_files = min(len(val) for key,val in dataset.items())
    
    min_idx = 0
    # list of random numbers (0, max_files, len=req_files)
    for folder, image_paths in dataset.items():
        max_files = len(image_paths)
        index_list = random.sample(range(0, max_files), req_files)
        for i in index_list:
            img = cv2.imread(dataset[folder][i])
            img = cv2.resize(img, (150, 150))
            training_dataset.append((img, folder))
    features = []
    label = []
    for feature, image_label in training_dataset:
        features.append(feature)
        label.append(data_directory.index(image_label))

    X_train = np.array(features)
    Y_train = np.array(label)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, shuffle=True)
    save_label = open("plain_map_model.pickle", "wb")
    pickle.dump(data_directory, save_label)
    save_label.close()
    return X_test, Y_test, X_train, Y_train


def get_model(X_test, Y_test, X_train, Y_train, epochs, lrate):
    # Sets the value of image dimension ordering convention with tensorflow backend - 'tf'
    global test_accuracy, training_accuracy
    # keras.backend.set_image_dim_ordering('tf')
    # Fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # Normalize the input data from 0-255 to 0-1.0
    X_train = X_train / 255.0
    # one hot encode outputs
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_train.shape[1]
    model = model_nn(num_classes, epochs, lrate)
    callbacks = [keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                             write_grads=False,
                                             write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                                             embeddings_metadata=None)]    

    os.system("rm -r ./logs")
    # Fit the model
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=32, shuffle=True, callbacks=callbacks,
                        validation_data=(X_test, Y_test))
    
    print(model.summary)
    return history['acc'], history['loss'], history['val_acc'], history['val_loss']


def model_nn(num_classes, epochs, lrate):
    model = Sequential()
    model.add(
        Conv2D(64, (3, 3), input_shape=(150, 150, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    decay = lrate / epochs
    sgd = SGD(lr = lrate, momentum = 0.2, decay = decay, nesterov = False)
    model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def comparison_results():
    ########### This can be done in a for loop but was facing issues and did not want to waste much time on it ##########
    epochs = 100
    X_test, Y_test, X_train, Y_train = load_map_centric_dataset()

    training_accuracy_average, validation_accuracy_average, training_loss_average, validation_loss_average = get_model(
        X_test, Y_test, X_train, Y_train, epochs=epochs, lrate=0.065)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(np.arange(0, epochs, 1), training_accuracy_average, 'g')
    axs[0, 0].plot(np.arange(0, epochs, 1), validation_accuracy_average, 'b')
    axs[0, 0].plot(np.arange(0, epochs, 1), training_loss_average, 'r')
    axs[0, 0].plot(np.arange(0, epochs, 1), validation_loss_average, 'y')
    axs[0, 0].set_title('Map Centric Dataset with epoch = {0}, lrate = 0.065'.format(epochs))
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].set_ylabel('accuracy / loss')


    X_test, Y_test, X_train, Y_train = load_robot_centric_dataset()

    training_accuracy_average, validation_accuracy_average, training_loss_average, validation_loss_average = get_model(
        X_test, Y_test, X_train, Y_train, epochs=epochs, lrate=0.065)

    axs[0, 1].plot(np.arange(0, epochs, 1), training_accuracy_average, 'g')
    axs[0, 1].plot(np.arange(0, epochs, 1), validation_accuracy_average, 'b')
    axs[0, 1].plot(np.arange(0, epochs, 1), training_loss_average, 'r')
    axs[0, 1].plot(np.arange(0, epochs, 1), validation_loss_average, 'y')
    axs[0, 1].set_title('Robot Centric Dataset with epoch = {0}, lrate = 0.065'.format(epochs))
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('accuracy / loss')


    fig.legend(['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'], loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    # X_test, Y_test, X_train, Y_train = load_robot_centric_dataset()
    # get_model(X_test, Y_test, X_train, Y_train, epochs=100, lrate=0.065)
    comparison_results()


if __name__ == "__main__":
    main()
