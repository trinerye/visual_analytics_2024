# Assignment 2: Classification benchmarks with Logistic Regression and Neural Networks

## Description of the assignment
This project uses scikit-learn to classify images in the Cifar10 dataset, which are preprocessed (greyscaled, normalized, and reshaped) before classification. 

In the ``src`` folder, you will find two Python scripts: one for training the logistic regression classifier and another for training the neural network classifier. Each script produces a classification report, saved in the ``out`` folder, while the trained models are saved in the ``models`` folder."You can also find an additional plot of the loss curve, which illustrates the neural network's performance, in the ``out`` folder. 

## Installation

 1. Open a terminal and clone the repository using Git 
```sh
git clone https://github.com/trinerye/visual_analytics_2024.git
```

2. Change directory to the assignment folder 
```sh
cd assignment_1
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
source ./A1_env/bin/activate
# Run the code
python src/logistic_regression.py &
python src/neural_network.py 
# Deactivate the enviroment
deactivate
```
