import os
import cv2
import numpy as np
import pandas as pd
import argparse
import sys
sys.path.append("..")
from utils.imutils import jimshow as show


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",
                        "-i",
                        type= str,
                        required = True,
                        help="Filename of the chosen image, fx:'image_0001.jpg'")
                        
    
    parser.add_argument("--print_results",
                        "-p",
                        action="store_true",
                        help="Saves the most similar images to the chosen image in the out folder if the flag is added.")

    return parser.parse_args()


# This function reads the chosen image and creates a normalized color histogram of it.
def create_chosen_hist(chosen_image_path):  
    chosen_image = cv2.imread(chosen_image_path)
    chosen_image_hist = cv2.calcHist([chosen_image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
    chosen_norm_hist = cv2.normalize(chosen_image_hist, chosen_image_hist, 0, 1.0, cv2.NORM_MINMAX)
    return chosen_norm_hist


# This function reads all the images, creates the corresponding color histograms and normalizes them.
# Afterwards it compares each normalized histogram with that of the chosen image and saves the image name and
# result in the appropriate list
def create_and_compare_hist(in_folderpath, image_files, chosen_norm_hist):

    image_names = []
    results = []

    for image_file in image_files:
        image_path = os.path.join(in_folderpath, image_file)
        image_names.append(image_file)
        image = cv2.imread(image_path)
        image_hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])
        norm_hist = cv2.normalize(image_hist, image_hist, 0, 1.0, cv2.NORM_MINMAX)
        comparisons = round(cv2.compareHist(chosen_norm_hist, norm_hist, cv2.HISTCMP_CHISQR), 2)
        results.append(comparisons)
    return image_names, results

def save_results(in_folderpath, out_folderpath, image_names, results, args):

    # Saves each image name and its corresponding result into a dataframe
    df = pd.DataFrame({'Filename': image_names,'Distance': results})

    # Sorts the dataframe by "distance" saving the lowest six results 
    df_sorted = df.sort_values(by=['Distance'], ascending=True).head(6)

    # Iterates over the elements in df_sorted, creates a folderpath for each image using the filename column, 
    # reads it and shows it
    for i, row in df_sorted.iterrows():
        sorted_image_names = row['Filename']
        sorted_image_path = os.path.join(in_folderpath, sorted_image_names)
        sorted_image = cv2.imread(sorted_image_path)

        # If the print_results flag is added then save the similar images to the 'out' folder and print a message to the screen
        if args.print_results:
  
            cv2.imwrite(os.path.join(out_folderpath, sorted_image_names), sorted_image)

            print(f"Saving {sorted_image_names} in the 'out' folder")
        
        else:
        # If not then print a message to the screen explaining how to add the flag
            print(f"To show the most similar images to '{args.image}', ensure you include the '--print_results' flag when executing this script.")
            break

    # Saves the sorted dataframe as a csv file in the out folder 
    df_sorted.to_csv(os.path.join(out_folderpath, 'image_comparisons.csv'), index=False)

def main():

    # Calls the parser function
    args = parser()

    # Creates 'out' folderpath
    out_folderpath = os.path.join("out")

    # If the directory does not exist, make the directory
    os.makedirs(out_folderpath, exist_ok=True)

    # Creates 'in' folderpath
    in_folderpath = os.path.join("in")

    # Creates a sorted list of all the directories within the given folder path
    image_files = sorted(os.listdir(in_folderpath))

    # Folderpath of the chosen image
    chosen_image_path = os.path.join(in_folderpath, args.image)

    # Calling the function which creates and normalizes the chosen image histogram
    chosen_norm_hist = create_chosen_hist(chosen_image_path)

    # Calling the function which compares the chosen images with the entire image dataset
    image_names, results = create_and_compare_hist(in_folderpath, image_files, chosen_norm_hist)

    # Calling the function which saves the images_names and results in a dataframe and turns it int a CSV
    save_results(in_folderpath, out_folderpath, image_names, results, args)

if __name__ == "__main__":
    main()

