# Assignment 3: Transfer learning with pretrained CNNs

## About

This project uses transfer learning with the pre-trained ``VGG16`` model to classify images from the ``Tobacco3482`` dataset, comprising scanned black-and-white images across ten classes. Additionally, the ``ImageDataGenerator`` from ``tensorflow`` augments the data due to the small size of some classes in the Tobacco3482 dataset.

However, before classification, the dataset is split into a train-and-test set using ``scikit-learn``, and the model is modified by removing its existing classification layers and adding custom layers using tensorflow, compiling the layers, and adding labels for the different categories. 

The ``src`` directory contains two scripts:

-  **transfer_learning.py:** Trains the CNN and generates a classification report.

- **plotting_tools.py:** Plots the loss curve.


### Data

Download the [Tobacco3482](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) dataset from Kaggle and save it in the ``in`` directory. However, the download contains a second Tobacco3482 directory within it, so be sure to remove that before running the code to prevent errors.

### Model

For this project, the [VGG16](https://keras.io/api/applications/vgg/) model is loaded without the top classification layers, marking the remaining layers as nontrainable while adding the following layers to enhance model performance. 

```sh
# New classification layers

flat1 = Flatten()(model.layers[-1].output)
bn = BatchNormalization()(flat1)
drop = Dropout(0.1)(bn)
class1 = Dense(128, activation='relu')(drop)
output = Dense(10, activation='softmax')(class1)
```
Afterwards, the model is compiled using the ``Adam`` optimizer with an ``ExponentialDecay()`` learning rate that fits the optimizer. The loss function is set to ``categorical_crossentropy`` with ``accuracy`` used as the evaluation metric.

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
        ├── readme.md
        ├── requirements.txt
        ├── run.sh
        └── setup.sh
```
## Usage

If you want replicate this project, follow the steps outlined below. The instructions will guide you through setting up the environment, running the script, and plotting the results while helping you understand the available command-line options for customizing the training process. 

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
cd assignment_3
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
python src/transfer_learning

# Deactivate the enviroment
deactivate
```

### Command Line Interface  

This project supports several command-line flags which customizes the training process. *See table for reference.*

|Flag      |Shorthand|Description                                |Type|Required|
|----------|---------|-------------------------------------------|----|--------|
| --epochs | -e      |Number of epochs you want the model to run |int |TRUE    |
| --print  | -p      |Saves the model summary in the outfolder   |str |FALSE   |

## Results 

Write something here

### Loss curve

![plot](out/loss_curve.png)

Describe what you see

### Limitations and future improvements 

Complex model architecture, exstensive fintuning --> Grid search



