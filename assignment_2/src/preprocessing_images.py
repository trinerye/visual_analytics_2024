import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.datasets import cifar10


def load_data():
    # Loads the cifar10 dataset
    print("Loading data")
    return cifar10.load_data()

def preprocess_images(images, labels): 

    image_list = []  
    
    # Iterates over each image in images 
    for image in tqdm(images, desc="Preprocessing image"):

        # Converts the images into greyscale
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        # Scales the images
        image_scaled = image_grey / 255.0 

        # Flattens the images
        image_flattened = image_scaled.flatten() 

        # Appends the flattened images to the image_list
        image_list.append(image_flattened) 

    # Converts the list of flattened images to a np array
    images_processed = np.array(image_list) 
    
    # Flattens the labels
    labels_processed = labels.flatten() 
    
    return images_processed, labels_processed

