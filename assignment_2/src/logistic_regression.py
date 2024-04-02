import os
import numpy as np
import cv2

# Machine learning stuff
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib
from joblib import dump

from tensorflow.keras.datasets import cifar10

# This function loads the data from the cifar10 dataset
def load_data():
    return cifar10.load_data()

# This function preprocesses the images
def preprocess_images(images):
    image_list = []  # Empty list which stores the flattened images
    
    for image in images:
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converts the images into greyscale
        image_scaled = image_grey / 255.0 # Scales the images
        image_flattened = image_scaled.flatten()  # Flattens the images
        image_list.append(image_flattened)  # Appends the flattened images to the image_list
    images_processed = np.array(image_list) # Converts the list of flattened images to a np array
    return images_processed

# This function trains the logistic regression model
def train_model(X_train_processed, y_train_processed):
    return LogisticRegression(tol=0.1, solver='saga', multi_class='multinomial', random_state=42).fit(X_train_processed, y_train_processed)

# This function evaluates the performance of the trained classifier on the test dataset and produces a classification report
def evaluate_model(y_test_processed, X_test_processed, classifier, labels):
    return metrics.classification_report(y_test_processed, classifier.predict(X_test_processed), target_names=labels)

# This function saves the classification report and the logistic regression classifier model
def saving_report(classifier_metrics, classifier, report_path, classifier_path):

    # Opens the file in the out folder in write mode and writes the classification metrics to it.
    with open(report_path, "w") as file:
        file.write(classifier_metrics)

    # Saves the trained classifier in the models folder
    joblib.dump(classifier, classifier_path)

def main():

    # Creates a filepath for each directory 
    out_folderpath = os.path.join("out", "logistic_regression")
    models_folderpath = os.path.join("models", "logistic_regression")

    # If the directory does not exist, make the directory
    os.makedirs(out_folderpath, exist_ok=True)
    os.makedirs(models_folderpath, exist_ok=True)

    # Filepath for each saved file
    classifier_path = os.path.join(models_folderpath, "regression_classifier.joblib")
    report_path = os.path.join(out_folderpath, "classification_report.txt")
  
    # Loading the data
    (X_train, y_train), (X_test, y_test) = load_data()

    # List of labels
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Preprocessing the training and test images
    X_train_processed = preprocess_images(X_train)
    X_test_processed = preprocess_images(X_test)
    y_train_processed = y_train.flatten()
    y_test_processed = y_test.flatten()

    #Training the logistic regression classifier
    classifier = train_model(X_train_processed, y_train_processed)

    # Creating the classification report
    classifier_metrics = evaluate_model(y_test_processed, X_test_processed, classifier, labels)

    # Saving the classification report and the logistic regression classifier model
    saving_report(classifier_metrics, classifier, report_path, classifier_path)

if __name__ == "__main__":
    main()