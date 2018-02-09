from matplotlib import pyplot as plt
from numpy import argmax
import numpy as np
from PIL import Image
import glob
def showImage(image_data):
    plt.imshow(image_data)
    plt.show()

def one_hot_to_categorical(one_hot):
    # result = argmax(predictions[0])
    return argmax(one_hot)

def shuffle_wrt_to_each_other(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# print(y_train.shape)
# print(y_train[0])
def print_clothing_article_class(index):
    # nums = [1, 2, 3, 4, 5]
    clothing_articles = ["T-shirt", "Trousers", "Pullover", "Dress", "coat",
    "Sandals", "Shirt", "Sneakers", "Bag", "ankle boot"]
    print("The image contains a " + clothing_articles[index])
    # return clothing_articles[index]

def print_image_class(index):
    Classes = ["Bird", " Plane", "Orange"]
    print("The image contains a " + Classes[index])

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_and_preprocess_images(path):
    locations = glob.glob(path)
    x = np.array([np.array(Image.open(fname).resize((100,100))) for fname in locations])
    gray_x =rgb2gray(x)
    return gray_x
