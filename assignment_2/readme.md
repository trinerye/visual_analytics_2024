# Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks

## About

This project uses ``scikit-learn`` to classify images from the ``Cifar-10`` dataset which have been greyscaled, normalized, and reshaped to fit the requirements of the ``LogisticRegression()`` and ``MLPClassifier()`` classifiers. 

The ``src`` directory contains two scripts:

-  **logistic_regression.py:** Trains the logistic regression classifier and saves a classification report in the ``out`` directory

- **neural_network.py:** Trains the neural network classifier and saves a classification report and a plot of the loss curve in the ``out`` directory

- **preprocessing_images.py:** Preprocesses the images from the cifar-10 dataset

Additionally, the trained models are stored in the ``models`` directory to ensure reproducibility.

### Data

The cifar-10 dataset consits of 60000 32x32 colour images divided into 10 categories. For more information about the cifar-10 dataset see here

You do not need a data folder for this repository as the cifar-10 dataset can be downloaded from the ``tensorflow.keras.datasets`` as follows:

```sh
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```
### Model

``LogisticRegression()`` and ``MLPClassifier()``

##  File Structure

```
└── assignment_2
        |
        ├── models
        │   ├── insert here
        │   └── insert here
        │      
        ├── out
        │   ├── logistic_regression_classification_report.txt
        |   ├── neural_network_classification_report.txt
        |   └── loss_curve.png
        |
        ├── src
        │   ├── logistic_regression.py
        │   ├── neural_network.py
        │   └── preprocess_images.py
        │     
        ├── readme.md
        ├── requirements.txt
        ├── run.sh
        └── setup.sh
```

## Usage

To run this project, follow the steps outlined below. These instructions will guide you through setting up the environment, preprocessing the cifar-10 dataset images, and training the logistic regression and neural network classifier. 

### Pre-Requisites

*Please makes sure to install the following requirements before running the script.*

**Python**: version 3.12.3

### Installation

**1.** Clone the repository using Git.
```sh
git clone https://github.com/trinerye/visual_analytics_2024.git
```

**2.** Change directory to the assignment folder.
```sh
cd assignment_2
```

**3.** Run ``setup.sh`` to install the dependencies needed for this project. 
```sh
bash setup.sh
```
**4.** Run ``run.sh`` to activate the environment and run the main script. 
```sh
bash run.sh
```

```sh
...
# Activate the environment (Unix/macOS)
source ./A3_env/bin/activate

# Run the code
python src/logistic_regression.py &
python src/neural_network.py

# Deactivate the enviroment
deactivate
```

## Results 

Write something here

### Loss curve

![plot](out/neural_network_loss_curve.png)

Describe what you see

### Limitations

Noget med at modellen er trænet på color images og derfor måske har svært ved at classify greyscaled images

### Future improvements 

Grid search