
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from PIL import Image
import utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import glob
bird_locations = glob.glob('/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/grayscale_birds/*.jpg')
plane_locations = glob.glob('/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/grayscale_planes/*.jpg')
orange_locations = glob.glob('/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/grayscale_oranges/*.jpg')

bird_x = np.array([np.array(Image.open(fname)) for fname in bird_locations])
plane_x = np.array([np.array(Image.open(fname)) for fname in plane_locations])
orange_x = np.array([np.array(Image.open(fname)) for fname in orange_locations])

bird_y = np.empty(bird_x.shape[0])
bird_y.fill(0)
plane_y = np.empty(plane_x.shape[0])
plane_y.fill(1)
orange_y = np.empty(orange_x.shape[0])
orange_y.fill(2)

x = np.concatenate((bird_x, plane_x, orange_x))
y = np.concatenate((bird_y, plane_y, orange_y))

# shuffle and split into training and test data sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30,random_state=42)

#Explicitly adding the second parameter (depth)
X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
#
# Conver to floats so the normalization division by max value works as you would expect
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize
# Normally datasets values range from 0 to 255. By dividing by max we get values
# between 0 and 1. neato.
X_train /= 255
X_test /= 255

# Convert our label to one-hot encoding
# vectors with the correct class having 1 and the rest having 0
Y_train = np_utils.to_categorical(Y_train,3)
Y_test = np_utils.to_categorical(Y_test, 3)

model = Sequential()

# First input layer. A lot going on here, so let me explain.
# First argument is how many kernel filters to use.(wtf is a kernel filter btw?)
# Second two int tuple defines the X and Y sizes of the kernels
# Activation defines the activation function
# input shape defines the X and Y sizes of our input images and 1 defines the depth
# depth is colour channels. Here we just have one, but for rgb it would be 3
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(100,100,1)))

model.add(Convolution2D(32, (3,3), activation='relu'))
# Pooling layer. Remember how we just find the largest values of some part?
# This is it. Reduces amount of parameters of our network
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout is for regulization. This disable some neurons some of the time so
# the model "Learns" ho to get good results even without part of itself
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch =2, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)


# serialize model to JSON
model_json = model.to_json()
with open("BPO100_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("BPO100_model.h5")
print("Saved model to disk")
