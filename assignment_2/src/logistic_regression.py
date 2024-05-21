import os
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib
from joblib import dump
from preprocessing_images import load_data, preprocess_images

# This function initializes the logistic regression classifier and fits it to the training data
def train_model(X_train_processed, y_train_processed):
    classifier = LogisticRegression(tol=0.1, solver='saga', multi_class='multinomial', random_state=42)
    classifier.fit(X_train_processed, y_train_processed)
    return classifier

# This function evaluates the performance of the trained classifier on the test dataset and produces a classification report
def evaluate_model(y_test_processed, X_test_processed, classifier, labels):    
    return metrics.classification_report(y_test_processed, classifier.predict(X_test_processed), target_names=labels)

# This function saves the classification report and the logistic regression classifier model
def saving_report(classifier_metrics, classifier, report_path, classifier_path):
        
    # Opens the file in the out folder in write mode and writes the classification metrics to it.
    with open(report_path, "w") as file:
        file.write(classifier_metrics)

    # Saves the trained classifier in the models folder
    joblib.dump(classifier, classifier_path)

def main():

    # Creates a filepath for each directory 
    out_folderpath = os.path.join("out")
    models_folderpath = os.path.join("models")

    # If the directory does not exist, make the directory
    os.makedirs(out_folderpath, exist_ok=True)
    os.makedirs(models_folderpath, exist_ok=True)

    # Creates a filepath for each saved file
    classifier_path = os.path.join(models_folderpath, "logistic_regression_classifier.joblib")
    report_path = os.path.join(out_folderpath, "logistic_regression_classification_report.txt")
  
    # Loads the data
    print("Loading data")
    (X_train, y_train), (X_test, y_test) = load_data()
 
    # List of labels
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Preprocesses the training and test images and labels 
    print("Processing images and labels")
    X_train_processed, y_train_processed = preprocess_images(X_train, y_train) 
    X_test_processed, y_test_processed = preprocess_images(X_test, y_test) 

    # Trains the logistic regression classifier
    print("Training the logistic regression classifier")
    classifier = train_model(X_train_processed, y_train_processed)

    # Creates the classification report
    print("Evaluating the logistic regression classifier")
    classifier_metrics = evaluate_model(y_test_processed, X_test_processed, classifier, labels)

    # Saves the classification report and the logistic regression classifier model
    print("Saving the classifation report and the logistic regression classifier")
    saving_report(classifier_metrics, classifier, report_path, classifier_path)

if __name__ == "__main__":
    main()

    