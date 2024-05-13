
# base tools
import os, sys
sys.path.append(os.path.join(".."))

# data analysis
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import random

from sklearn.neighbors import NearestNeighbors

# tensorflow
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import (load_img, 
                                                  img_to_array)
from tensorflow.keras.applications.vgg16 import (VGG16, 
                                                 preprocess_input)
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def extract_features(in_folderpath, model):

    feature_list = []

    filepath = [in_folderpath + "/" + file for file in sorted(os.listdir(in_folderpath)) if file.endswith('jpg')]

    for i in tqdm(range(len(filepath))):
        
    # dirs = sorted(os.listdir(in_folderpath))

    # test_dirs = random.choices(dirs, k=6)

    # for file in test_dirs:   

        # filepath = os.path.join(in_folderpath, file)
         
        # if file.endswith('jpg'):
            
        image = load_img(filepath[i], target_size=(224, 224))
        image_array = img_to_array(image)

        expanded_image_array = np.expand_dims(image_array, axis=0)

        image_processed = preprocess_input(expanded_image_array)

        # print(image_processed)

        features = model.predict(image_processed, verbose=False)
        # print(features)

        # # flatten
        flattened_features = features.flatten()
        # # print(flattened_features)

        # normalized_features = flattened_features / norm(features)

        normalized_features = flattened_features/255.0
        
        feature_list.append(normalized_features)


    return filepath, np.array(feature_list)

def load_model():
    return VGG16(weights='imagenet', 
              include_top=False,
              pooling='avg',
              input_shape=(224, 224, 3))

def main():

    in_folderpath = os.path.join("in", "flowers")

    out_folderpath = os.path.join("out", "knn_images")
    os.makedirs(out_folderpath, exist_ok=True)

    model = load_model()

    filepath, feature_list = extract_features(in_folderpath, model)

    neighbors = NearestNeighbors(n_neighbors=10, 
                                algorithm='brute',
                                metric='cosine').fit(feature_list)

    distances, indices = neighbors.kneighbors([feature_list[250]]) # add argparse here

    idxs = []
    for i in range(1,6):
        print(distances[0][i], indices[0][i])
        idxs.append(indices[0][i])
    
    # plt target
    plt.imshow(mpimg.imread(filepath[250]))

    # plot 3 most similar
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(mpimg.imread(filepath[idxs[0]]))
    axarr[1].imshow(mpimg.imread(filepath[idxs[1]]))
    axarr[2].imshow(mpimg.imread(filepath[idxs[2]]))

    # Save the plot
    plot_path = os.path.join(out_folderpath, 'image_comparison_plot.png')
    plt.savefig(plot_path)
    plt.close()  


if __name__ == "__main__":
    main()

