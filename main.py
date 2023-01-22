from classifiers import hyperparameter_tuning, model_validation, tuning_results
from dataset import load_data, correlation_matrix
from feature_selection import FeatureSelector
import matplotlib.pyplot as plt

########################### FEATURE SELECTION ##################################
FEATURE_SELECTOR = "PCA" # PCA - LDA
########################### CLASSIFIERS ########################################
BAYES = True     # Naive Bayes
KNN = True      # K Nearest Neighbors
SVM = True      # Support Vector Machine
NN = True       # Neural Network
KM = True       # K Means

def main():
    ################ DATASET PREPARATION #######################################

    x_train, y_train, x_test, y_test = load_data(verbose=True)
    
    ############# FEATURES CORRELATION MATRIX ##################################
    correlation_matrix()

    ########## FEATURE SELECTION ###############################################

    feature_selector = FeatureSelector(FEATURE_SELECTOR)
    x_train = feature_selector.fit_transform(x_train, y_train.values.ravel())
    x_test = feature_selector.transform(x_test)
    
    ################## HYPERPARAMETER TUNING AND MODEL TRAINING ################

    trained_models = hyperparameter_tuning(FEATURE_SELECTOR, BAYES, KNN, KM, SVM, NN, x_train, y_train)
    # Show the results of the tuning
    tuning_results(trained_models)

    ################## MODEL VALIDATION ########################################

    model_validation(trained_models, x_test, y_test)

if __name__ == "__main__":
    main()