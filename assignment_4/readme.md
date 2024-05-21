# Assignment 4: Detecting faces in historical newspapers

## About

This project uses transfer learning with the pre-trained ``VGG16`` model to classify images from the ``Tobacco3482`` dataset, comprising scanned black-and-white images across ten categories. Additionally, the ``ImageDataGenerator`` from ``tensorflow`` augments the data due to the small size of some categories in the Tobacco3482 dataset.

Before the classification, however, the dataset is split into a train-and-test set using ``scikit-learn``, and the model gets modified by removing its existing classification layers and adding custom layers using tensorflow, compiling the layers, and adding labels for the different categories.  

The ``src`` directory contains two scripts:

- **face_detection.py:** 

- **plotting_tools.py:** Plots the results 


### Data

Download the ``Tobacco3482`` dataset from Kaggle here and save it in the ``in`` directory. Be aware that the download from Kaggle contains a second Tobacco3482 directory within it, so be sure to remove that to prevent errors when running the code. 

### Model

For this project, the VGG16 model is loaded without the top classification layers, marking the remaining layers as nontrainable while adding the following layers to enhance model performance. 


Afterwards, the model is compiled using the ``Adam`` optimizer with an ``ExponentialDecay()`` learning rate at ``0.001`` to fit the optimizer. The loss function is set to ``categorical_crossentropy`` with ``accuracy`` used as the evaluation metric.

##  File Structure

```
└── assignment_4
        |
        ├── in
        │   └── newspapers
        |         ├── GDL
        |         ├── IMP      
        |         └── JDG  
        |
        ├── out
        |   ├── classification_report.txt
        |   └── loss_curve.png
        |
        ├── src
        │   ├── face_detection.py
        │   └── plotting_tools.py
        │     
        ├── readme.md
        ├── requirements.txt
        ├── run.sh
        └── setup.sh
```
## Usage

To run this project, follow the steps outlined below. These instructions will guide you through setting up the environment, running the script, saving and plotting the results while helping you understand the available command-line options for customizing the training process. 

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
cd assignment_4
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
source ./A4_env/bin/activate

# Run the code
python src/face_detection.py -p 

# Deactivate the enviroment
deactivate
```

### Command Line Interface  

This project supports several command-line flags to customize the training process. *See table for reference.*

|Flag      |Shorthand|Description                                |Type|Required|
|----------|---------|-------------------------------------------|----|--------|
| --print  | -p      |Saves the model summary in the outfolder   |str |FALSE   |

## Results 

Write something here

### Loss curve

![plot](out/loss_curve.png)

Describe what you see

### Limitations

Noget med at modellen er trænet på color images og derfor måske har svært ved at classify greyscaled images

### Future improvements 

Grid search



