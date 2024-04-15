# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD, Adam

#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt
from plotting_tools import plot_history

# general
import os

def load_and_process_images(in_folderpath):

   dirs = sorted(os.listdir(in_folderpath))

   images = []
   labels = []

   for i, directory in enumerate(dirs):   

      subdir = sorted(os.listdir(os.path.join(in_folderpath, directory)))
      
      for file in subdir:

         filepath = os.path.join(in_folderpath, directory, file)
         
         if file.endswith('jpg'):
            
            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image_processed = preprocess_input(image)
            images.append(image_processed)
            labels.append(i)

   return np.array(images), np.array(labels)

def process_labels(y_train, y_test):
   lb = LabelBinarizer()
   y_train_encoded = lb.fit_transform(y_train)
   y_test_encoded = lb.fit_transform(y_test)
   return y_train_encoded, y_test_encoded   

def setup_model():

   tf.keras.backend.clear_session()

   # load model without classifier layers
   model = VGG16(include_top=False, 
               pooling='avg',
               input_shape=(224, 224, 3))

   # mark loaded layers as not trainable
   for layer in model.layers:
      layer.trainable = False

   # model.summary()

   # Add new classifier layers
   flat1 = Flatten()(model.layers[-1].output)
   class1 = Dense(128, activation='relu')(flat1)
   output = Dense(10, activation='softmax')(class1)

   # Define new model
   model = Model(inputs=model.inputs, outputs=output)

   # model.summary()
   return model 

def compile_model(model):
   
   # Compile
   lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.01,
      decay_steps=10000,
      decay_rate=0.9)
   sgd = SGD(learning_rate=lr_schedule)

   model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

   return model 

def train_model(model, X_train, y_train):
   # Train the model
   H = model.fit(X_train, y_train, 
            validation_split=0.1,
            batch_size=128,
            epochs=10,
            verbose=1)
   return H

def evaluate_model(model, X_test, y_test, label_names):

   predictions = model.predict(X_test, batch_size=128)

   classifier_metrics = classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names)
   return classifier_metrics

def save_report(classifier_metrics, report_path):
   with open(report_path, "w") as file:
        file.write(classifier_metrics)
        
def main():

   out_folderpath = os.path.join("out")

   # If the directory does not exist, make the directory
   os.makedirs(out_folderpath, exist_ok=True)

   in_folderpath = os.path.join("in", "Tobacco3482")

   report_path = os.path.join(out_folderpath, "classification_report.txt")

   images, labels = load_and_process_images(in_folderpath)
      
   X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
   
   y_train, y_test = process_labels(y_train, y_test)
   
   label_names = ["ADVE", "Email", "Form", "Letter", "Memo", "News","Note", "Report", "Resume", "Scientific"]

   model = setup_model()

   model = compile_model(model)

   H = train_model(model, X_train, y_train)

   plot_history(H, 10)

   classifier_metrics = evaluate_model(model, X_test, y_test, label_names)

   save_report(classifier_metrics, report_path)

if __name__ == "__main__":
    main()


