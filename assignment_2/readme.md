# Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks

## About

This project uses ``scikit-learn`` to classify images from the ``Cifar-10`` dataset which have been greyscaled, normalized, and reshaped to fit the requirements of the ``LogisticRegression()`` and ``MLPClassifier()`` classifiers. 

There are two scripts in the ``src`` folder, one for training the logistic regression classifier and another for training the neural network classifier. Each script generates a detailed classification report stored in the ``out`` folder with an additional plot of the neural network classifier's loss curve illustrating its performance. Lastly, the trained models and the vectorizer are stored in the ``models`` folder to ensure reproducibility.

### Data

You do not need a data folder for this repository as the cifar-10 dataset can be downloaded from the ``tensorflow.keras.datasets`` as follows:

```sh
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```
Read more about the cifar-10 dataset here

### Model

Describe the models here

## Setup

###  :file_folder: File Structure

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
        ├── .gitignore
        ├── README.md
        ├── requirements.txt
        ├── run.sh
        └── setup.sh
```

###  :electric_plug: Installation

 1. Open a terminal and clone the repository using Git 
```sh
git clone https://github.com/trinerye/visual_analytics_2024.git
```

2. Change directory to the assignment folder 
```sh
cd assignment_2
```

3. Run the setup script to install opencv-python, pandas, and numpy. It simultaneously creates a virtual environment containing the specific versions used to develop this project. 
```sh
bash setup.sh
```

Write something about how it creates and environment 

### Pre-Requisites

the requirements file

### Commands

How to run it

```sh
bash run.sh
```
Activate the environment and run the main script. Be aware that it deactivates the environment again after running the  script.

```sh
# Activate the environment (Unix/macOS)
source ./A2_env/bin/activate
# Run the code
python src/logistic_regression.py &
python src/neural_network.py 
# Deactivate the enviroment
deactivate
```

## Usage

Write about how to use the flags

### Command Line Interface Arguments 

```
  -a, --AAA         Write a short description here
  -b, --BBB         Write a short description here
  -c, --CCC         Write a short description here
  -h, --help        Show help [boolean]

```

## Resources