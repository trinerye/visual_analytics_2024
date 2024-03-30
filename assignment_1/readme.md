# Assignment 1: Simple image search algorithm

## Description of the assignment
This project uses OpenCV and the ’17 Category Flower Dataset’ to create a simple image search algorithm, which compares a ‘target flower’ to the rest of the dataset. It does so by creating a normalized colour histogram of each flower, comparing it to that of the target flower, displaying the top five flowers and saving the results as a CSV file in the out folder. 


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
python src/image_search_algorithm.py --image "image_1357.jpg" --print_results 
# Deactivate the enviroment
deactivate
```
## Usage

Write something about the flags here

```sh

```
