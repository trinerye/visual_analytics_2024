import os
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image, ImageFile
from plot_function import plot_results

ImageFile.LOAD_TRUNCATED_IMAGES = True

def parser():

     # Creates an argparse object 
    parser = argparse.ArgumentParser()

    # Defines the CLI argument that creates and saves an unedited version of the csv (OPTIONAL)                    
    parser.add_argument("--print",
                        "-p",
                        action="store_true",
                        help="Include this flag to save an unedited version of the csv")

    return parser.parse_args()  # Parses and returns the CLI arguments


def extract_data(in_folderpath, directory, mtcnn):

    # Creates a dictionary to store the data
    data = {'Newspaper': [],'Year': [],'Month': [], 'Day': [],'Filename': [], 
                'Faces':[],'Face Detected': [],'Pages': []} 

    # Creates a sorted list of the content in each directory  
    subdir = sorted(os.listdir(os.path.join(in_folderpath, directory))) 

    # Itereates over each file in the directory
    for file in tqdm(subdir, desc="Processing images"):

        if file.endswith('jpg'):

            # Creates a filepath for each file
            filepath = os.path.join(in_folderpath, directory, file)

            # Extracts the metadata from the filename
            extract_metadata(file, data)

            # Detects how many faces each file has
            detect_faces(filepath, mtcnn, data)
   
    return data
  
def create_dataframe(out_folderpath, directory, data, args): 

    # Creates a directory in the out folder for each directory 
    directory_folderpath = os.path.join(out_folderpath, directory)
    os.makedirs(directory_folderpath, exist_ok=True)

    # Turns the dictionary containing the data into a dataframe 
    df = pd.DataFrame(data)

    # Converts the values in the year column to integers 
    df['Year'] = df['Year'].astype('int') 

    # Calculates the decade for each year and saves the information in a new column
    decade = np.round(df['Year']//10*10, decimals=0).astype('int') 
    df.insert(1, 'Decade', decade)

  # Saves an unedited version of the csv if the --print flag is added when running the script
    if args.print:
        df.to_csv(os.path.join(directory_folderpath, f"{directory}_newspaper.csv"), index=False)
    else: 
        print(f"Include the --print flag to save an unedited version of the csv")
    
    return directory_folderpath, df

def extract_metadata(file, data):
    
    # Splits the filename into seperate entities 
    split_file = file.split("-")

    # Adds each entity to the corresponding list in the data dictionary 
    data['Newspaper'].append(split_file[0])
    data['Year'].append(split_file[1])
    data['Month'].append(split_file[2])
    data['Day'].append(split_file[3])
    data['Filename'].append(split_file[5])
    data['Pages'].append(1) 

def detect_faces(filepath, mtcnn, data):

    # Loads the images
    img = Image.open(filepath)  
    
    # Detects faces in the images
    boxes, _ = mtcnn.detect(img)

    img.close()

    # If the model does not detect a face assign an empthy list to boxes and append the number 0 to the face detected column 
    if boxes is None:
        boxes = []
        data['Face Detected'].append(0) 
    else: 
        # If a face is detected append the number 1 to the face detected column 
        data['Face Detected'].append(1) 
        
    # Appends the length of the np.array containing the bounding boxes to the faces column
    data['Faces'].append(len(boxes))    

def calculate_percentages(directory, directory_folderpath, df):

    # Groups the dataframe by decade and sums the content in the three columns
    sorted_df = df.groupby('Decade')[['Faces', 'Face Detected', 'Pages']].sum()

    # Calculate the percentage of pages with faces for each decade and inserts this information in a new column
    percentage = np.round(sorted_df['Face Detected']/sorted_df['Pages'], decimals=2)
    sorted_df.insert(3, 'Pages with Faces (%)', percentage)

    # Saves the sorted dataframe as a csv file 
    sorted_df.to_csv(os.path.join(directory_folderpath, f"{directory}_sorted_newspaper_information.csv"))

    return sorted_df

def main():

    # Creates a filepath for each directory and makes the out directory if does not exist
    in_folderpath = os.path.join("in", "newspapers")
    out_folderpath = os.path.join("out")
    os.makedirs(out_folderpath, exist_ok=True)

    # Calls the parser function
    args = parser()

    # Initializes the MTCNN model for face detection
    mtcnn = MTCNN(keep_all=True)

    # Creates a sorted list of the content in the in folder 
    dirs = sorted(os.listdir(in_folderpath))

    # Iterates over each directory in the in folder
    for index, directory in enumerate(dirs):

        # Extracts the metadata and the faces detected and saves it in a dictionary   
        data = extract_data(in_folderpath, directory, mtcnn)

        # Converts the data into a dataframe and adds a decade column to it
        directory_folderpath, df = create_dataframe(out_folderpath, directory, data, args)

        # Calculates the percentages of pages with faces for each decade
        sorted_df = calculate_percentages(directory, directory_folderpath, df)

        # Plots the results for each directory
        plot_results(sorted_df, directory, directory_folderpath, index)

if __name__ == "__main__":
    main()


