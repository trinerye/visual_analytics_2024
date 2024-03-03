# Assignment 1: Simple image search algorithm

## Description of the assignment
This project uses OpenCV and the ’17 Category Flower Dataset’ to create a simple image search algorithm, which compares a ‘target flower’ to the rest of the dataset. It does so by creating a normalized colour histogram of each flower, comparing it to that of the target flower, displaying the top five flowers and saving the results as a CSV file in the out folder. 


## Installation

 1. Clone the repository using Git 
```sh
git clone https://github.com/trinerye/visual_analytics_2024.git
```

2. Change directory to the assignment folder 
```sh
cd assignment1
```

3. Before running the script make sure to install opencv-python, matplotlib, pandas, and numpy by running the setup.sh 
```sh
bash setup.sh
```
4. To run the code open assignment1.ipynb and run all
```sh
assignment1.ipynb
```

## Usage
The main function takes three parameters - the in_folderpath (the in folder),  the out_folderpath (the out folder), and an integer referring to the number of the chosen flower (*Important: the image_index starts at zero, so if you want to compare image number 1357 as I did, the argument given to the function is 1356). Also, if you structure the folders differently, remember to update the file paths accordingly.

```sh
def main(in_folderpath, out_folderpath, image_index):

if __name__ == "__main__":

 main("../in", "../out", 1356)
```
