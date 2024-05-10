import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_dataframe(in_folderpath, out_folderpath, directory, mtcnn):

    directory_folderpath = os.path.join(out_folderpath, directory)

    os.makedirs(directory_folderpath, exist_ok=True)

    # dictionary = {'Newspaper': [],
    #             'Year': [],
    #             'Month': [], 
    #             'Day': [], 
    #             'Filename': [], 
    #             'Detected Faces':[],
    #             'Pages with Faces': [],
    #             'Pages Total': []} 

    # subdir = sorted(os.listdir(os.path.join(in_folderpath, directory))) 

    # # test_subdir = random.choices(subdir, k=6)

    # for file in tqdm(subdir, desc="Processing image"):

    #     if file.endswith('jpg'):

    #         filepath = os.path.join(in_folderpath, directory, file)

    #         extract_metadata(file, dictionary)

    #         detect_faces(filepath, mtcnn, dictionary)

    # df = pd.DataFrame(dictionary)

    # df['Year'] = df['Year'].astype('int') 

    # df.to_csv(os.path.join(directory_folderpath, f"{directory}_newspaper.csv"), index=False)
    
    return directory_folderpath


def extract_metadata(file, dictionary):
    split_file = file.split("-")
    dictionary['Newspaper'].append(split_file[0])
    dictionary['Year'].append(split_file[1])
    dictionary['Month'].append(split_file[2])
    dictionary['Day'].append(split_file[3])
    dictionary['Filename'].append(split_file[5])
    dictionary['Pages Total'].append(1)  


def detect_faces(filepath, mtcnn, dictionary):
    # Load an image containing faces
    img = Image.open(filepath)

    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)

    img.close()

    # If the model does not detect a face assign an empthy list to the variable boxes
    if boxes is None:
        boxes = []
        dictionary['Pages with Faces'].append(0)
    
    else: 
        dictionary['Pages with Faces'].append(1)
        # Appends the length of the np.array containing the bounding boxes
    dictionary['Detected Faces'].append(len(boxes))    


def calculate_decade(directory, directory_folderpath):
    
    altered_df = pd.read_csv(os.path.join(directory_folderpath, f"{directory}_newspaper.csv")) 

    year_to_decade = np.round(altered_df['Year']//10*10, decimals=0)

    altered_df.insert(1, 'Decade', year_to_decade)

    return altered_df


def calculate_percentages(directory, directory_folderpath, altered_df):

    sorted_df = altered_df.groupby('Decade')[['Detected Faces', 'Pages with Faces', 'Pages Total']].sum()

    percentage = np.round(sorted_df['Pages with Faces']/sorted_df['Pages Total'], decimals=2)

    sorted_df.insert(3, 'Percentage of Pages with Faces', percentage)

    sorted_df.to_csv(os.path.join(directory_folderpath, f"{directory}_sorted_newspaper_information.csv"))

    return sorted_df


def plot_results(sorted_df, directory, directory_folderpath):
    plt.figure(figsize=(12, 8))
    sorted_df['Percentage of Pages with Faces'].plot(kind='bar') 
    plt.title('Percentage of Pages with Faces by Decade', weight='bold')  
    plt.ylabel('Percentage of Pages with Faces', weight='bold')  
    plt.xlabel('Decade', weight='bold')  
    plt.xticks(rotation=0)  
    plt.savefig(os.path.join(directory_folderpath, f"{directory}_distribution_across_decades.jpg")) 
    plt.close()


def main():

    # Creates a filepath for each directory
    in_folderpath = os.path.join("in")
    out_folderpath = os.path.join("out")
   
    # If the out directory does not exist, make the directory
    os.makedirs(out_folderpath, exist_ok=True)

    mtcnn = MTCNN(keep_all=True)

    dirs = sorted(os.listdir(in_folderpath))

    for directory in dirs:
   
        directory_folderpath = create_dataframe(in_folderpath, out_folderpath, directory, mtcnn)

        altered_df = calculate_decade(directory, directory_folderpath)

        sorted_df = calculate_percentages(directory, directory_folderpath, altered_df)

        plot_results(sorted_df, directory, directory_folderpath)

if __name__ == "__main__":
    main()


