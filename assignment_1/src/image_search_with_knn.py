
# base tools
import os
import argparse
from tqdm import tqdm

# data analysis
import pandas as pd
import numpy as np
# from numpy.linalg import norm

from sklearn.neighbors import NearestNeighbors

# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",
                        "-i",
                        type= int,
                        required = True,
                        help="Write the index of your chosen image e.g. '0' if you which to compare 'image_0001.jpg' to the rest of the dataset")
                        
    parser.add_argument("--print_results",
                        "-p",
                        action="store_true",
                        help="Saves the most similar images to the chosen image in the out folder if this flag is added.")

    return parser.parse_args()

def load_model():
    return VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(filepath, model):

    feature_list = []

    for i in tqdm(range(len(filepath))):
            
        image = load_img(filepath[i], target_size=(224, 224))
        
        image_array = img_to_array(image)

        expanded_image_array = np.expand_dims(image_array, axis=0)

        image_processed = preprocess_input(expanded_image_array)

        features = model.predict(image_processed, verbose=False)

        flattened_features = features.flatten()

        # normalized_features = flattened_features / norm(features)

        normalized_features = flattened_features/255.0
        
        feature_list.append(normalized_features)

    return np.array(feature_list)


def compare_features(feature_list, args):

    neighbors = NearestNeighbors(n_neighbors=10, # what does this parameter do?
                                algorithm='brute',
                                metric='cosine').fit(feature_list)
    
    distances, indices = neighbors.kneighbors([feature_list[args.index]]) 

    return distances, indices

def save_csv(distances, indices, filenames, out_folderpath):

    index = []
    dist = []
    files = []
    
    for i in range(6):

        index.append(indices[0][i])
        dist.append(distances[0][i])
        files.append(filenames[indices[0][i]])

    dictionary = {'Index': index, 'Filename': files, 'Distance': dist} 
    
    df = pd.DataFrame(dictionary)
     
    df.to_csv(os.path.join(out_folderpath, 'test.csv'), index = False)

    return index, files
    

def save_plot(filepath, args, index, files, out_folderpath):

    # plt target
    plt.imshow(mpimg.imread(filepath[args.index])) 

    ### Save the target image
    ### Save the images like the first assignment 

    f, axarr = plt.subplots(1, 6, figsize=(20, 5))  # Adjust figure size to your needs
    axarr[0].imshow(mpimg.imread(filepath[index[0]]))
    axarr[0].set_title(f"Target image: {files[0]}")
    axarr[0].axis('off')  # Hide axes

    for i in range(1,6):
        axarr[i].imshow(mpimg.imread(filepath[index[i]]))
        axarr[i].set_title(f"{files[i]}")
        axarr[i].axis('off')  #Hide axes

    # Save the plot
    plot_path = os.path.join(out_folderpath, 'image_comparison_plot.png')
    plt.savefig(plot_path)
    plt.close(f)  

def main():

    in_folderpath = os.path.join("in", "flowers")

    out_folderpath = os.path.join("out", "knn_images")

    os.makedirs(out_folderpath, exist_ok=True)

    args = parser()

    model = load_model()

    filenames = [file for file in sorted(os.listdir(in_folderpath)) if file.endswith('jpg')]

    filepath = [os.path.join(in_folderpath, file) for file in filenames]

    feature_list = extract_features(filepath, model)

    distances, indices = compare_features(feature_list, args)

    index, files = save_csv(distances, indices, filenames, out_folderpath)

    save_plot(filepath, args, index, files, out_folderpath)

    print(files)
    
if __name__ == "__main__":
    main()

