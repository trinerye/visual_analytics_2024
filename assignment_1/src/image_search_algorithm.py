import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        "-i",
                        type= str,
                        required = True,
                        help="Filename of the chosen image, e.g., 'image_0001.jpg'")
                        
    parser.add_argument("--print",
                        "-p",
                        action="store_true",
                        help="Saves the most similar images to the chosen image in the out folder if this flag is added.")

    return parser.parse_args()

# This function takes an image path as an argument and creates a normalized color histogram of it.
def create_histogram(image_path):  
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    norm_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    return norm_hist

def compare_histograms(in_folderpath, chosen_norm_hist):
    
    dirs = sorted(os.listdir(in_folderpath))

    filenames = []
    distances = []

    for image in tqdm(dirs):
        filenames.append(image)
        image_path = os.path.join(in_folderpath, image)
        norm_hist = create_histogram(image_path)
        comparison = round(cv2.compareHist(chosen_norm_hist, norm_hist, cv2.HISTCMP_CHISQR), 2)
        distances.append(comparison)
    return filenames, distances

def save_result(in_folderpath, out_folderpath, filenames, distances, args):

    # Saves each image name and its corresponding result into a dataframe
    df = pd.DataFrame({'Filename': filenames,'Distance': distances})

    # Sorts the dataframe by "distance" saving the lowest six results 
    sorted_df = df.sort_values(by=['Distance'], ascending=True).head(6)

    # Iterates over each filename in the dataframe and creates a folderpath for each image reads it and write it to the outfolder it
    
    for filename in sorted_df['Filename']:

        sorted_image_path = os.path.join(in_folderpath, filename)
        
        sorted_images = cv2.imread(sorted_image_path)

        # If the print_results flag is added then save the similar images to the 'out' folder and print a message to the screen
        if args.print:

            cv2.imwrite(os.path.join(out_folderpath, filename), sorted_images)

            print(f"Saving {filename} in the 'out' folder")
        
        else:
        # If not then print a message to the screen explaining how to add the flag
            print(f"To show the most similar images to '{args.image}', ensure you include the '--print_results' flag when executing this script.")
        

    # Saves the sorted dataframe as a csv file in the out folder 
    sorted_df.to_csv(os.path.join(out_folderpath, 'image_comparisons.csv'), index=False)

def main():

    # Calls the parser function
    args = parser()

    # Creates the folderpaths
    in_folderpath = os.path.join("in", "flowers")
    out_folderpath = os.path.join("out", "cv2_images")
    os.makedirs(out_folderpath, exist_ok=True)
    
    # Folderpath for the chosen image
    chosen_image_path = os.path.join(in_folderpath, args.image)

    # Passing the chosen_image_path as an argument to the create_histogram function
    chosen_norm_hist = create_histogram(chosen_image_path)

    # Calling the function which compares the chosen images with the entire image dataset
    filenames, distances = compare_histograms(in_folderpath, chosen_norm_hist)

    # Calling the function which saves the images_names and results in a dataframe and turns it int a CSV
    save_result(in_folderpath, out_folderpath, filenames, distances, args)

if __name__ == "__main__":
    main()


