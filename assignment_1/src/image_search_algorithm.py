import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def parser():

    # Creates an argparse object 
    parser = argparse.ArgumentParser()

    # Defines the CLI argument that specifies the filenames of the selected image (REQUIRED)
    parser.add_argument("--image",
                        "-i",
                        type=str,
                        required=True,
                        help="Specify the filename of the image you want to compare, e.g., image_0001.jpg")
    
    # Defines the CLI argument that creates and saves a plot of the results (OPTIONAL)                    
    parser.add_argument("--print",
                        "-p",
                        action="store_true",
                        help="Include this flag to create and save a plot showing the result")

    return parser.parse_args()  # Parses and returns the CLI arguments

def create_histogram(filepath):  

    # Reads the image 
    image = cv2.imread(filepath)
    
    # Creates a color histogram of that image
    hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])

    # Normalizes the color histogram
    norm_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    
    return norm_hist

def compare_histograms(in_folderpath, chosen_norm_hist):
   
   # Creates a sorted list of the content in the in folder 
    dirs = sorted(os.listdir(in_folderpath))

    filenames = []
    distances = []

    # Iterates over each file in the directory
    for file in tqdm(dirs, desc="Compares images"):

        # Adds the name of the image to a list
        filenames.append(file)

        # Creates a filepath for each image 
        filepath = os.path.join(in_folderpath, file)

        # Calls the create_histogram function to create a color histogram of all the images in the directory
        norm_hist = create_histogram(filepath)

        # Compares the selected image with all the other images and calculate the distance between them
        comparison = round(cv2.compareHist(chosen_norm_hist, norm_hist, cv2.HISTCMP_CHISQR), 2)

        # Adds the result to a list
        distances.append(comparison)

    return filenames, distances

def save_csv(in_folderpath, out_folderpath, filenames, distances, args):

    # Saves the filenames and the distances as a dataframe
    df = pd.DataFrame({'Filename': filenames,'Distance': distances})

    # Sorts the dataframe by "distance" and saves the top 6 results
    sorted_df = df.sort_values(by=['Distance'], ascending=True).head(6)

    # Saves the sorted dataframe as a csv file in the out folder 
    sorted_df.to_csv(os.path.join(out_folderpath, 'image_comparisons.csv'), index=False)

    return sorted_df

def save_plot(sorted_df, in_folderpath, out_folderpath):

    # Iterates over each filename in sorted_df and creates a folderpath for each image
    sorted_filepath = [os.path.join(in_folderpath, filename) for filename in sorted_df['Filename']]
    
    # Creates a figure with 6 subplots 
    fig, axarr = plt.subplots(1, 6, figsize=(20, 5))

    # Displays the selected image in the first subplot
    axarr[0].imshow(mpimg.imread(sorted_filepath[0]))
    axarr[0].set_title(f"$\\bf{{Selected\\ image}}$\n {sorted_df.iloc[0]['Filename']} \nDistance: {sorted_df.iloc[0]['Distance']}")
    axarr[0].axis('off')  

    # Iterates over the remaining 5 subplots, displaying the nearest neighbor images
    for i in range(1, 6):
        axarr[i].imshow(mpimg.imread(sorted_filepath[i]))
        axarr[i].set_title(f"{sorted_df.iloc[i]['Filename']} \nDistance: {sorted_df.iloc[i]['Distance']:.4f}")
        axarr[i].axis('off') 

    # Saves the plot in the outfolder and closes the figure
    plot_path = os.path.join(out_folderpath, 'image_comparison_plot.png')
    plt.savefig(plot_path)
    plt.close(fig)  

def main():

    # Calls the parser function
    args = parser()

    # Creates the folder paths
    in_folderpath = os.path.join("in", "flowers")
    out_folderpath = os.path.join("out", "cv2_images")
    os.makedirs(out_folderpath, exist_ok=True)
    
    # Filepath for the selected image
    chosen_image_path = os.path.join(in_folderpath, args.image)

    # Creates a histogram for the selected image.
    chosen_norm_hist = create_histogram(chosen_image_path)

    # Compares the selected image with the entire dataset
    filenames, distances = compare_histograms(in_folderpath, chosen_norm_hist)

    # Saves the results as a csv file
    sorted_df = save_csv(in_folderpath, out_folderpath, filenames, distances, args)

    # Creates and saves the plot if the --print flag is added when running the script
    if args.print:
        print(f"Saving plot in the 'out' folder")
        save_plot(sorted_df, in_folderpath, out_folderpath)
    else:
       # If not it prints a message to the screen explaining how to add the flag 
        print(f"Include the --print flag to create and save a plot showing the result")

if __name__ == "__main__":
    main()
