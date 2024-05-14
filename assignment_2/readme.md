# Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks

This project uses ``scikit-learn`` to classify images from the ``Cifar-10`` dataset, which consists of 60000 images divided into a predefined train/test split. However, before classification, the images are preprocessed (greyscaled, normalized, and reshaped) to fit the requirements of the ``LogisticRegression()`` and ``MLPClassifier()`` classifiers.
You do not need a data folder for this repository as the cifar-10 dataset can be downloaded from the ``tensorflow.keras.datasets`` the following way:

```sh
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```
See here to read more about the cifar-10 dataset.

In the ``src`` folder, you will find two scripts: one for training the logistic regression classifier and another for training the neural network classifier. Each script also produces a classification report, saved in the ou``out`` folder, with an additional plot of the loss curve illustrating the neural network classifier's performance. Also, the trained models and vectorizer can be found in the ``models`` folder to ensure reproducibility. 


## Installation

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

4. Activate the environment and run the main script. Be aware that it deactivates the environment again after running the  script.
```sh
bash run.sh
```
```sh
# Activate the environment (Unix/macOS)
source ./A2_env/bin/activate
# Run the code
python src/logistic_regression.py &
python src/neural_network.py 
# Deactivate the enviroment
deactivate
```
