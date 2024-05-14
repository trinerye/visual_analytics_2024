import os
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib
from joblib import dump
from process_images import load_data, preprocess_images

# This function trains the logistic regression model
def train_model(X_train_processed, y_train_processed):
    classifier = MLPClassifier(hidden_layer_sizes = (128,), max_iter=1000, random_state = 42, early_stopping=True)
    classifier.fit(X_train_processed, y_train_processed)
    return classifier

# This function evaluates the performance of the trained classifier on the test dataset and produces a classification report
def evaluate_model(y_test_processed, X_test_processed, classifier, labels):
    return metrics.classification_report(y_test_processed, classifier.predict(X_test_processed), target_names=labels)
    
def plot_loss_curve(classifier, plot_path):
    plt.plot(classifier.loss_curve_)
    plt.title("Loss curve during training", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Loss score')
    plt.savefig(plot_path)
    
# This function saves the classification report and the logistic regression classifier model
def saving_report(classifier_metrics, classifier, report_path, classifier_path):

    # Opens the file in the out folder in write mode and writes the classification metrics to it.
    with open(report_path, "w") as file:
        file.write(classifier_metrics)
    
    # Saves the trained classifier in the models folder
    joblib.dump(classifier, classifier_path)

def main():
   
    # Creates a filepath for each directory 
    out_folderpath = os.path.join("out", "neural_network")
    models_folderpath = os.path.join("models", "neural_network")

    # If the directory does not exist, make the directory
    os.makedirs(out_folderpath, exist_ok=True)
    os.makedirs(models_folderpath, exist_ok=True)

    # Filepath for each saved file
    classifier_path = os.path.join(models_folderpath, "neural_network_classifier.joblib")
    report_path = os.path.join(out_folderpath, "classification_report.txt")
    plot_path = os.path.join(out_folderpath, "loss_curve.png")
  
    # Loading the data
    (X_train, y_train), (X_test, y_test) = load_data()

    # List of labels
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Preprocessing the training and test images
    X_train_processed, y_train_processed = preprocess_images(X_train, y_train)
    X_test_processed, y_test_processed = preprocess_images(X_test, y_test)

    #Training the neural network
    print("Training the neural network classifier") 
    classifier = train_model(X_train_processed, y_train_processed)

    # Creating the classification report
    print("Evaluating the neural network classifier")
    classifier_metrics = evaluate_model(y_test_processed, X_test_processed, classifier, labels)

    # Plotting the loss curve and saving it 
    print(f"Saving the loss curve at {plot_path}")
    plot_loss_curve(classifier, plot_path)

    # Saving the classification report and the neural network model
    print("Saving the classifation report and the neural network classifier")
    saving_report(classifier_metrics, classifier, report_path, classifier_path)

if __name__ == "__main__":
    main()