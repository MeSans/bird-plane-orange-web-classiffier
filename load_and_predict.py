from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json
import utils
from matplotlib import pyplot as plt
from numpy import argmax
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
#-------------------load iamge data for evaluation
import glob
# bird_locations = glob.glob(
#     '/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/grayscale_birds/*.jpg')
# plane_locations = glob.glob(
#     '/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/grayscale_planes/*.jpg')
# orange_locations = glob.glob(
#     '/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/grayscale_oranges/*.jpg')

bird_x = utils.load_and_preprocess_images('/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/dumped_birds/*.jpg')
plane_x = utils.load_and_preprocess_images('/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/airplanes_raw/*.jpg')
orange_x = utils.load_and_preprocess_images('/home/patriks/Desktop/bird-plane-orange-web-classiffier/datasets/oranges_raw/*.jpg')
# bird_x = np.array([np.array(Image.open(fname)) for fname in bird_locations])
# plane_x = np.array([np.array(Image.open(fname)) for fname in plane_locations])
# orange_x = np.array([np.array(Image.open(fname))
                     # for fname in orange_locations])

bird_y = np.empty(bird_x.shape[0])
bird_y.fill(0)
plane_y = np.empty(plane_x.shape[0])
plane_y.fill(1)
orange_y = np.empty(orange_x.shape[0])
orange_y.fill(2)

x = np.concatenate((bird_x, plane_x, orange_x))
y = np.concatenate((bird_y, plane_y, orange_y))

X_train, X_test, Y_train, Y_test = train_test_split(x, y,
                                                test_size=0.30, random_state=42)
utils.showImage(X_test[250])

X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
#
#
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# print(X_train[0])
#
# Normalize
# Normally datasets values range from 0 to 255. By dividing by max we get values
# between 0 and 1. neato.
X_train /= 255
X_test /= 255

# Convert our label to one-hot encoding
# vectors with the correct class having 1 and the rest having 0
Y_train = np_utils.to_categorical(Y_train, 3)
Y_test = np_utils.to_categorical(Y_test, 3)
#------------------------------------
# load json and create model
json_file = open('BPO100_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("BPO100_model.h5")
print("Loaded model from disk")

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# utils.showImage(X_test[1])
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# Squish data into a range from 0 to 1
# X_test /= 255

# Convert our label to one-hot encoding
# vectors with the correct class having 1 and the rest having 0
# y_test = np_utils.to_categorical(y_test, 3)
# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

predictions = loaded_model.predict(X_test)
result = utils.one_hot_to_categorical(predictions[250])
print(result)
utils.print_image_class(result)

# Results are in the on-hot encoded format,
# predictions = loaded_model.predict(X_test)
# So we convert back to categorical and see the result
# result = utils.one_hot_to_categorical(predictions[1])
# print(predictions[1])
# print(result)
