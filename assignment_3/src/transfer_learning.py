import os
import numpy as np
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from plotting_tools import plot_history

def parser():

   # Creates an argparse object 
   parser = argparse.ArgumentParser()

   # Defines the CLI argument that creates and saves a model summary in the outfolder (OPTIONAL)
   parser.add_argument("--epochs",
                     "-e",
                     type=int,
                     required=True,
                     help="Specify the amount of epochs you want for the model e.g. 20")
   
   # Defines the CLI argument that creates and saves a model summary in the outfolder (OPTIONAL)
   parser.add_argument("--print",
                     "-p",
                     action="store_true",
                     help="Include this flag to save a model summary in the outfolder")
    
   return parser.parse_args()  # Parses and returns the CLI arguments


def load_and_process_images(in_folderpath):
   
   # Creates a sorted list of the content in the in folder 
   dirs = sorted(os.listdir(in_folderpath))
   
   images = []
   labels = []

   # Iterates over the directories in the in directory 
   for i, directory in enumerate(dirs):

      # Creates a sorted list of filepaths for each file in the directories if the image is a jpg
      filepaths = [os.path.join(in_folderpath, directory, file) for file in sorted(os.listdir(os.path.join(in_folderpath, directory))) if file.endswith('jpg')]
      
      # Iterates over each filepath in the list of filepaths
      for filepath in tqdm(filepaths, desc="Processing images"):

         # Loads the image, resizing it to the target size
         image = load_img(filepath, target_size=(224, 224))

         # Converts the image to a np.array
         image = img_to_array(image)
         
         # Preprocesses the images
         image_processed = preprocess_input(image)

         # Appends the processed images and labels to their corresponding lists
         images.append(image_processed)
         labels.append(i)

   return np.array(images), np.array(labels) # Returns the lists as np.arrays

def process_labels(y_train, y_test):

   # Initializes the LabelBinarizer obejct
   lb = LabelBinarizer()

   # Fits the LabelBinarizer obejct to the labels and transform them to a one-hot encoding
   y_train = lb.fit_transform(y_train)
   y_test = lb.fit_transform(y_test)

   return  y_train, y_test

def setup_model(args):
   
   # Clears the memory 
   tf.keras.backend.clear_session()

   # Loads the model without the classification layers
   model = VGG16(include_top=False, pooling='avg', input_shape=(224, 224, 3))

   # Marks the loaded layers as not trainable
   for layer in model.layers:
      layer.trainable = False

   # Adds new classification layers to the model
   flat1 = Flatten()(model.layers[-1].output)
   bn = BatchNormalization()(flat1)
   class1 = Dense(128, activation='relu')(bn)
   drop = Dropout(0.1)(class1)
   output = Dense(10, activation='softmax')(drop)

   # Defines the new model
   model = Model(inputs=model.inputs, outputs=output)

   # If the argparse flag --print is add, print the model summary to the screen
   if args.print:
      model.summary()

   return model 

def compile_model(model):
   
   # Initializes the learning_rate_schedule 
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, 
                                                                decay_steps=10000,
                                                                decay_rate=0.9)
   # Initializes the adam optimizer 
   adam = Adam(learning_rate=lr_schedule)

   # Compiles the final model 
   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
   
   return model 

def generate_data():

   # Uses the ImageDataGenerator to generate more data e.g. data augmentation
   datagen = ImageDataGenerator(horizontal_flip=True, 
                                rotation_range=20,
                                fill_mode='nearest',
                                brightness_range=[0.8,1.2],
                                validation_split=0.1)
   return datagen

def train_model(datagen, model, X_train, y_train, args):   

   # Fits the model on the training and validation data using data augmentation
   H = model.fit(datagen.flow(X_train, y_train, batch_size=128), 
                              validation_data = datagen.flow(X_train, y_train, 
                                                             batch_size=128, 
                                                             subset = "validation"),
                                                             epochs=args.epochs,
                                                             verbose=1) 

   return H

def evaluate_model(model, X_test, y_test):

   # List of labels
   label_names = ["ADVE", "Email", "Form", "Letter", "Memo", "News","Note", "Report", "Resume", "Scientific"]

   # Generates predictions 
   predictions = model.predict(X_test, batch_size=128)

   # Creates a classification report
   classifier_metrics = classification_report(y_test.argmax(axis=1),
                                              predictions.argmax(axis=1),
                                              target_names=label_names)

   return classifier_metrics

def save_report(classifier_metrics, out_folderpath):

   # Creates the classification report folderpath
   report_path = os.path.join(out_folderpath, "classification_report.txt")

   # Creates a txt file and writes the content of the classification report to it
   with open(report_path, "w") as file:
        file.write(classifier_metrics)
        
def main():

   # If the directory does not exist, make the directory
   in_folderpath = os.path.join("in", "Tobacco3482")
   out_folderpath = os.path.join("out")
   os.makedirs(out_folderpath, exist_ok=True)

   # Calls the parser function
   args = parser()
   
   # Loads and preprocesses the images from the tobacco3482 dataset
   images, labels = load_and_process_images(in_folderpath)
   
   # Splits the dataset into a train and test set.
   X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
   
   # Preprocesses the labels to fit the model
   y_train, y_test = process_labels(y_train, y_test)

   # Initilizes the model with new classification layers
   model = setup_model(args)

   # Initializes the learning_rate_schedule, the optimizer and compiles the final model
   model = compile_model(model)

   # Generates data using data augmentation
   datagen = generate_data()

   # Trains the model
   H = train_model(datagen, model, X_train, y_train, args)

   # Plots the loss and accuracy curves and saves them in the out folder
   plot_history(H, args.epochs, out_folderpath) 

   # Evaluates the model, creates a classification report
   classifier_metrics = evaluate_model(model, X_test, y_test)

   # Saves the classification report
   save_report(classifier_metrics, out_folderpath)

if __name__ == "__main__":
    main()


