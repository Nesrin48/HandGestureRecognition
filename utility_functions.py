# Here we import everything we need for the project
# It is always recomended to put them in the first cell
# Helps present results as a confusion-matrix
from sklearn.metrics import confusion_matrix
# Helps with organizing data for training
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from google.colab import drive
import os


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # for one hot encoding
from keras.models import load_model  # to load the saved model
import json  # to load the json file
from PIL import Image  # Image manipulations


# To open the drive


# Import of keras model and hidden layers for our convolutional network

# Sklearn

# Set all the parameter of the model here
num_classes = 36

# This function is used more for debugging and showing results later. It plots the image into the notebook


def imshow(image_path):
    # Open the image to show it in the first column of the plot
    image = Image.open(image_path)
    # Create the figure
    fig = plt.figure(figsize=(50, 5))
    ax = fig.add_subplot(1, 1, 1)
    # Plot the image in the first axe with it's category name
    ax.axis('off')
    ax.set_title(image_path)
    ax.imshow(image)


def get_hand_gesture_labels(image_dir):
    """
    Creates a dictionary of image labels (results_dic) based upon the filenames 
    of the image files. These image image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the image in the image.
    Be sure to format the image labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'hand1_z_dif_seg_4_cropped.png' image label = 'z')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = image image label (string)
    """

    # Retrieve the filenames from folder hand_gesture_images/
    in_files = os.listdir(image_dir)
    # Creates empty dictionary for the results (image labels, etc.)
    results_dic = dict()
    # Processes through each file in the directory, extracting only the words
    # of the file that contain the image image label
    for idx in range(0, len(in_files), 1):
        # Skips file if starts with . (like .DS_Store of Mac OSX) because it
        # isn't an hand gesture image file
        if in_files[idx][0] != ".":
            if in_files[idx] not in results_dic:
                # Creates temporary label variable to hold pet label name extracted
                hand_gesture_label = ""
                # Sets hand_gesture_image variable to a filename after removing the file extension => we can escape this step :D.
                hand_gesture_image = os.path.splitext(in_files[idx])[0]
                # Sets string to lower case letters
                low_hand_gesture_image = hand_gesture_image.lower()
                # Split the file name by "_" splitter
                word_list_hand_gesture_image = low_hand_gesture_image.split(
                    "_")
                # get hand gesture element which is in the position 2 in the file name
                hand_gesture_name = word_list_hand_gesture_image[1]

                # Strip off starting/trailing whitespace characters
                hand_gesture_label = hand_gesture_name.strip()

                # Add hand_gesture_label as value of the file name key
                results_dic[in_files[idx]] = hand_gesture_label
            else:
                print("** Warning: Duplicate files exist in directory:",
                      in_files[idx])

    return results_dic


def load_dataset(image_dir, results_dic):
    X = []  # Image data
    Y = []  # Labels

    # Loops through imagepaths to load images and labels into arrays
    for key in results_dic:
        # Reads image and returns np.array
        img = cv2.imread(image_dir + '/' + key)
        # Converts into the corret colorspace (GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reduce image size so training can be faster
        img = cv2.resize(img, (320, 120))
        X.append(img)

        # Get the lable
        label = results_dic[key]
        # Append the lables
        Y.append(label)

    # Turn X and y into np.array to speed up train_test_split
    X = np.array(X, dtype="uint8")
    Y = np.array(Y)
    print("len(image_dir)", len(image_dir))
    print("len(image_dir)", len(results_dic))
    print("Shape of X ", X.shape)
    print("Shape of Y ", Y.shape)
    # Needed to reshape so CNN knows it's different images
    X = X.reshape(len(results_dic), 120, 320, 1)
    print("Shape of X after reshaping ", X.shape)
    return X, Y

# Function to split the dataset into Train and Test set


def split_dataset(X, Y):
    # Percentage of images that we want to use for testing. The rest is used for training.
    ts = 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=ts, random_state=42)
    return X_train, X_test, y_train, y_test

# Add one hote encoding for lables


def one_hot_encode(lables):
    label_encoder = LabelEncoder()
    vec = label_encoder.fit_transform(lables)
    return vec


# Construction of model
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    return model


# trin the model
def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=5,
    batch_size=64,
    verbose=2
):

    # Configures the model for training
    model.compile(optimizer='adam',  # Optimization routine, which tells the computer how to adjust the parameter values to minimize the loss function.
                  # Loss function, which tells us how bad our predictions are.
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # List of metrics to be evaluated by the model during training and testing.
    # Trains the model for a given number of epochs (iterations on a dataset) and validates it.
    model.fit(X_train, y_train, epochs=5, batch_size=64,
              verbose=2, validation_data=(X_test, y_test))


# Test The Model
def test_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy: {:2.2f}%'.format(test_acc*100))

# Save entire model to a HDF5 file


def save_model(path, model):
    if not os.path.exists(path):
        print('save directories...', flush=True)
        os.makedirs(path)
    model.save(path + '/handrecognition_model.h5')


def load_cnn_model(path):
    model = load_model(path + '/handrecognition_model.h5')
    return model

# Label mapping


def cat_to_name(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

# Image Preprocessing


def process_image(image_path):
    X = []  # Image data
    img = cv2.imread(image_path)  # Reads image and returns np.array
    # Converts into the corret colorspace (GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reduce image size so training can be faster
    img = cv2.resize(img, (320, 120))
    X.append(img)
    # Turn X into np.array to speed up train_test_split
    X = np.array(X, dtype="uint8")
    print("Shape of X ", X.shape)
    # Needed to reshape so CNN knows it's different images
    X = X.reshape(len(X), 120, 320, 1)
    print("Shape of X after reshaping ", X.shape)
    return X

# Class Prediction


def predict(image_path, model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    X = process_image(image_path)
    # Get the topk prob using model.predict_proba which return top probablilties
    # for all calsses in descendent order
    top_p = model.predict_proba(X[:1])[0]
    # Here we get the topk probas but in assedent order
    top_p = [top_p[i] for i in np.argsort(top_p)[-topk:]]
    # revers the order to get the bigest value to the smallest one in the list of top probs
    top_p = top_p[::-1]
    # Get the topk classes
    top_c = model.predict(X[:1])
    top_c = np.argsort(top_c)[:, -topk:][0]
    top_c = top_c[::-1]  # reverse elment to get the descenedent order

    # Extract the classes names from classes indices
    top_classes = [
        cat_to_name[str(category)] for category in top_c
    ]
    return top_p, top_classes

# Sanity Checking


def display_prediction(image_path, model, cat_to_name, topk=5):
    # Open the image to show it in the first column of the plot
    image = Image.open(image_path)
    # predict topk classes
    top_probs, top_classes = predict(image_path, model, cat_to_name, topk)
    # Get the top class that have the max probat
    top_p = top_probs[np.argmax(top_probs)]  # the top class proba
    top_c = top_classes[np.argmax(top_probs)]  # top class label
    print("###\nThe top {} classes with their probas for the given input ###\n".format(topk))
    for result in range(len(top_probs)):
        print('Rank=> {:<2}| Class=> {:<4}| Proba=> {:.4f}'.format(result + 1, top_classes[result], top_probs[result]))
  
    print("\nThe top class of the input is => {} \n".format(top_c))