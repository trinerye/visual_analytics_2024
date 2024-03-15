# Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks

## Description of the assignment
This project uses scikit-learn to classify images in the Cifar10 dataset, which are preprocessed (greyscaled, normalized, and reshaped) before classification. 

In the ´´src´´ folder you will find two python scripts: one for training the logistic regression classifier and another which trains the neural network classifier. Each script produces a classification reports, which are saved in the out folder, but the loss curve plot only applies to the neural network. Each trained model is saved in the ´´models´´ folder.  

## Installation

 1. Clone the repository using Git
```sh
git clone https://github.com/trinerye/visual_analytics_2024.git
```

2. Change directory to ´´src´´ in the assignment folder 
```sh
cd assignment_2/src
```

3. Before running the script make sure to install opencv-python, matplotlib, numpy,scikit-learn and tensorflowby in the requirements.txt file by running the setup.sh 
```sh
bash setup.sh
```

4. To execute the code run the python script in the terminal
```sh
python logistic_regression.py
python neural_network.py 
```
