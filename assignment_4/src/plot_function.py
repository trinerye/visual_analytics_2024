import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_results(sorted_df, directory, directory_folderpath, index):

    # List containing the color for each plot
    colors = ['#eb6859', '#fc8862', '#fda26b']

    # Creates a barplot for each directory showing the percentage of pages with faces and saves it in the out folder
    plt.figure(figsize=(12, 8))
    sorted_df['Pages with Faces (%)'].plot(kind='bar', color=colors[index]) 
    plt.title('Percentage of Pages with Faces by Decade', weight='bold')  
    plt.ylabel('Pages with Faces (%)', weight='bold')  
    plt.xlabel('Decade', weight='bold')  
    plt.xticks(rotation=0)  
    plot_folderpath = os.path.join(directory_folderpath, f"{directory}_distribution_across_decades.jpg")
    plt.savefig(plot_folderpath) 
    plt.close()

