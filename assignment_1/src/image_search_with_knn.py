import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def parser():

    # Creates an argparse object 
    parser = argparse.ArgumentParser()

    # Defines the CLI argument that specifies the image index for the selected image (REQUIRED)
    parser.add_argument("--index",
                        "-i",
                        type= int,
                        required = True,
                        help="Specify the index of the image you want to compare, e.g., '0' for the first image in the dataset (image_0001.jpg)")

    # Defines the CLI argument that creates and saves a plot of the results (OPTIONAL)
    parser.add_argument("--print",
                        "-p",
                        action="store_true",
                        help="Include this flag to create and save a plot showing the result")
    
    return parser.parse_args()  # Parses and returns the CLI arguments

def load_model():
    
    # Initializing the VGG16 model
    return VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(filepath, model):

    feature_list = []

# Iterates through the list of filepaths 
    for i in tqdm(range(len(filepath)), desc="Processing images"):
        
        # Loads and reshapes the image data
        image = load_img(filepath[i], target_size=(224, 224))

        # Converts the image into a numpy array.
        image_array = img_to_array(image)

        # Expands the array by adding an extra dimension to it
        expanded_image_array = np.expand_dims(image_array, axis=0)

        # Preprocesses the array to fit the VGG16 model
        image_processed = preprocess_input(expanded_image_array)

        # Uses the model to extract features from the preprocessed images
        features = model.predict(image_processed, verbose=False)

        # Flattens the extracted features into a 1D array
        flattened_features = features.flatten()

        # Normalizes the flattened features by dividing them with the maximum pixel value
        normalized_features = flattened_features/255.0
        
        # Appends the normalized features to the feature list
        feature_list.append(normalized_features)
    
    return np.array(feature_list) # Returns the feature list as a np.array


def compare_features(feature_list, args):

    #  Initializing the NearestNeighbors model to find the six nearest neighbors to the selected image 
    neighbors = NearestNeighbors(n_neighbors= 6,
                                algorithm='brute',
                                metric='cosine').fit(feature_list)

    # Returns the distances and indices of the 6 nearest neighbors to the selected image
    distances, indices = neighbors.kneighbors([feature_list[args.index]]) 

    return distances, indices

def save_csv(distances, indices, filenames, out_folderpath):

    index = []
    dist = []
    files = []
    
    # Iterate over the 6 nearest neighbors
    for i in range(6):

        # Appends the index, the distance, and the filename for each neighbor to the corresponding list
        index.append(indices[0][i])
        dist.append(distances[0][i])
        files.append(filenames[indices[0][i]])

    # Creates a dictionary containing the index, filenames, and distances for each neighbor
    dictionary = {'Index': index, 'Filename': files, 'Distance': dist} 
    
    # Converts the dictionary into a pandas dataframe 
    df = pd.DataFrame(dictionary)
    
    # Saves the dataframe as a csv in the out folder
    df.to_csv(os.path.join(out_folderpath, 'knn_image_comparisons.csv'), index = False)

    return index, files, dist

def save_plot(filepaths, args, index, files, dist, out_folderpath):
    # Creates a figure with 6 subplots 
    fig, axarr = plt.subplots(1, 6, figsize=(18, 5))

    # Displays the selected image in the first subplot
    axarr[0].imshow(mpimg.imread(filepaths[index[0]]))
    axarr[0].set_title(f"$\\bf{{Selected\\ image}}$\n {files[0]} \nDistance: {dist[0]}")
    axarr[0].axis('off')  

    # Iterates over the remaining 5 subplots, displaying the nearest neighbor images
    for i in range(1,6):
        axarr[i].imshow(mpimg.imread(filepaths[index[i]]))
        axarr[i].set_title(f"{files[i]} \nDistance: {dist[i]:.4f}")
        axarr[i].axis('off') 

    # Saves the plot in the outfolder and closes the figure
    plot_path = os.path.join(out_folderpath, 'knn_image_comparison_plot.png')
    plt.savefig(plot_path)
    plt.close(fig)  

def main():
    # Creates the folder paths
    in_folderpath = os.path.join("in", "flowers")
    out_folderpath = os.path.join("out", "knn_images")
    os.makedirs(out_folderpath, exist_ok=True)

    # Creates filenames and filepaths for each image
    filenames = [file for file in sorted(os.listdir(in_folderpath)) if file.endswith('jpg')]
    filepaths = [os.path.join(in_folderpath, file) for file in filenames]

    # Calls the parser function
    args = parser()

    # Loads the VGG16 model
    model = load_model()

    # Extracts features from all the images 
    feature_list = extract_features(filepaths, model)

    # Compares the extracted features and finds the nearest neighbors for the selected image
    distances, indices = compare_features(feature_list, args)
    
    # Saves the result as a csv in the out folder
    index, files, dist = save_csv(distances, indices, filenames, out_folderpath)

    # Creates and saves the plot if the --print flag is added when running the script
    if args.print:
        print(f"Saving plot in the 'out' folder")
        save_plot(filepaths, args, index, files, dist, out_folderpath)
    else:
        # If not it prints a message to the screen explaining how to add the flag 
        print(f"Include the -print flag to create and save a plot showing the result")

if __name__ == "__main__":
    main()

