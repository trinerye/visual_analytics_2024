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
                     help="Specify the amount of epochs you want for the model e.g. 12")
   
   # Defines the CLI argument that creates and saves a model summary in the outfolder (OPTIONAL)
   parser.add_argument("--print",
                     "-p",
                     action="store_true",
                     help="Include this flag to save a model summary in the outfolder")
    
   return parser.parse_args()  # Parses and returns the CLI arguments


def load_and_process_images(in_folderpath):
  
   dirs = sorted(os.listdir(in_folderpath))
   
   images = []
   labels = []

   for i, directory in enumerate(dirs):

      filepaths = [os.path.join(in_folderpath, directory, file) for file in sorted(os.listdir(os.path.join(in_folderpath, directory))) if file.endswith('jpg')]
      
      for filepath in tqdm(filepaths, desc="Processing images"):

         image = load_img(filepath, target_size=(224, 224))
         image = img_to_array(image)
         image_processed = preprocess_input(image)
         images.append(image_processed)
         labels.append(i)

   return np.array(images), np.array(labels)

def process_labels(images, labels):

   X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

   lb = LabelBinarizer()

   y_train = lb.fit_transform(y_train)
   y_test = lb.fit_transform(y_test)

   return  X_train, X_test, y_train, y_test

def setup_model(args):

   tf.keras.backend.clear_session()

   # load model without classifier layers
   model = VGG16(include_top=False, pooling='avg', input_shape=(224, 224, 3))

   # mark loaded layers as not trainable
   for layer in model.layers:
      layer.trainable = False

   # Add new classifier layers
   flat1 = Flatten()(model.layers[-1].output)
   bn = BatchNormalization()(flat1)
   drop = Dropout(0.1)(bn)
   class1 = Dense(128, activation='relu')(drop)
   output = Dense(10, activation='softmax')(class1)

   # Define new model
   model = Model(inputs=model.inputs, outputs=output)

   if args.print:
      model.summary()

   return model 

def compile_model(model):
   
   # Compile - check the best learning rate/parameters for the adam optimizer
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, 
                                                                decay_steps=10000,
                                                                decay_rate=0.9)
   
   # sgd = SGD(learning_rate=lr_schedule) use argparse to choose the optimizer  # arg
   adam = Adam(learning_rate=lr_schedule)

   model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
   
   return model 

def generate_data():

   datagen = ImageDataGenerator(horizontal_flip=True, 
                                rotation_range=20,
                                fill_mode='nearest',
                                brightness_range=[0.8,1.2],
                                validation_split=0.1)
   return datagen

def train_model(datagen, model, X_train, y_train, args):   

   # fits the model on batches with real-time data augmentation:
   H = model.fit(datagen.flow(X_train, y_train, batch_size=128), 
                              validation_data = datagen.flow(X_train, y_train, 
                                                             batch_size=128, 
                                                             subset = "validation"),
                                                             epochs=args.epochs,
                                                             verbose=1) 

   return H

def evaluate_model(model, X_test, y_test):

   label_names = ["ADVE", "Email", "Form", "Letter", "Memo", "News","Note", "Report", "Resume", "Scientific"]

   predictions = model.predict(X_test, batch_size=128)

   classifier_metrics = classification_report(y_test.argmax(axis=1),
                                              predictions.argmax(axis=1),
                                              target_names=label_names)

   return classifier_metrics

def save_report(classifier_metrics, out_folderpath):

   report_path = os.path.join(out_folderpath, "classification_report.txt")

   with open(report_path, "w") as file:
        file.write(classifier_metrics)
        
def main():

   # If the directory does not exist, make the directory
   in_folderpath = os.path.join("in", "Tobacco3482")
   out_folderpath = os.path.join("out")
   os.makedirs(out_folderpath, exist_ok=True)

   # Calls the parser function
   args = parser()
  
   images, labels = load_and_process_images(in_folderpath)
   
   X_train, X_test, y_train, y_test = process_labels(images, labels)

   model = setup_model(args)

   model = compile_model(model)

   datagen = generate_data()

   H = train_model(datagen, model, X_train, y_train, args)

   plot_history(H, args.epochs, out_folderpath) # argparse here

   classifier_metrics = evaluate_model(model, X_test, y_test)

   save_report(classifier_metrics, out_folderpath)

if __name__ == "__main__":
    main()


