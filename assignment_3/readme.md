# Assignment 3: Transfer learning with pretrained CNNs

## About

This project uses transfer learning with the pre-trained ``VGG16`` model to classify images from the ``Tobacco3482`` dataset, comprising scanned black-and-white images across ten categories. Additionally, the ``ImageDataGenerator`` from ``tensorflow`` augments the data due to the small size of some categories in the Tobacco3482 dataset.

Before the classification, however, the dataset is split into a train-and-test set using ``scikit-learn``, and the model gets modified by removing its existing classification layers and adding custom layers using tensorflow, compiling the layers, and adding labels for the different categories.  

The src folder contains two scripts:

-  **transfer_learning.py:** Trains the CNN and generates a classification report.

- **plotting_tools.py:** Plots the loss curve.


### Data

Download the ``Tobacco3482`` dataset from Kaggle here and save it in the ``in`` folder. Be aware that the downloaded folder from Kaggle contains a second Tobacco3482 folder within it, so be sure to remove that from the folder to prevent errors when running the code. 

### Model

The VGG16 model is loaded without the top classification layers, marking the remaining layers as nontrainable while adding the following layers to enhance model performance. 

```
# New classification layers

flat1 = Flatten()(model.layers[-1].output)
bn = BatchNormalization()(flat1)
drop = Dropout(0.1)(bn)
class1 = Dense(128, activation='relu')(drop)
output = Dense(10, activation='softmax')(class1)
```
Afterwards, the model is compiled using the ``Adam`` optimizer with an ``ExponentialDecay()`` learning rate at ``0.001`` to fit the optimizer. The loss function is set to ``categorical_crossentropy`` with ``accuracy`` used as the evaluation metric.

##  File Structure

```
└── assignment_3
        |
        ├── in
        │   └── Tobacco3482 (contains 3482 files)
        │      
        ├── out
        |   ├── classification_report.txt
        |   └── loss_curve.png
        |
        ├── src
        │   ├── plotting_tools.py
        │   └── transfer_learning.py
        │     
        ├── .gitignore
        ├── README.md
        ├── requirements.txt
        ├── run.sh
        └── setup.sh
```
## Usage

### Pre-Requisites

Please makes sure to install the following requirements before running the script.

**Python**: version 3.12.3

### Installation

 Open a terminal terminal and clone the repository using Git 
```sh
git clone https://github.com/trinerye/visual_analytics_2024.git
```

Change directory to the assignment folder 
```sh
cd assignment_3
```

### Commands

Run the setup script to install the dependencies needed for this project. It simultaneously creates a virtual environment containing the specific versions used to develop this project. 
```sh
bash setup.sh
```

Activate the environment and run the main script. Be aware that it deactivates the environment again after running the  script.
```sh
bash run.sh
```
```sh
# Activate the environment (Unix/macOS)
source ./A3_env/bin/activate
# Run the code
python src/transfer_learning
# Deactivate the enviroment
deactivate
```

### Command Line Interface  

Write about how to use the flags

|Flag     |Shorthand|Description                 |Type|Required|
|---------|---------|----------------------------|----|--------|
|--epochs |-e       |Write something here        |int |TRUE    |
|--print  |-p       |Write something here        |str |FALSE   |

## Discussion 

### Summary of the key points from the outputs 

### Limitations

### Future improvements 



