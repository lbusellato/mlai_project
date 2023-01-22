import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from dataclasses import dataclass
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from utils import pop_substrings, plot_confusion_matrix, plot_tuning_results, model_comparison

@dataclass
class Classifier:
    clf : GridSearchCV
    name : str

# False Alarm Rate score
def FAR(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    far = fp / (fp + tn)
    return far

# Hyperparameter tuning
def hyperparameter_tuning(FEATURE_SELECTOR, BAYES, KNN, KM, SVM, NN, x_train, y_train):
    # List  of models to tune and train. Each model is specified as a dictionary:
    #
    # model { 'type'   : Model type,
    #         'name'   : Name to save the model as,
    #         'use'    : Boolean that enables/disables the use of the model,
    #         'params' : Dict of ranges for hyperparameter tuning }
    #
    fsel = FEATURE_SELECTOR + '_'
    models = [{'type' : GaussianNB(), 'name' : fsel + 'Bayes', 'use' : BAYES, 'params' : {}},
            {'type' : KNeighborsClassifier(n_jobs=-1), 'name' : fsel + 'KNN', 'use' : KNN, 'params' : {'n_neighbors' : np.linspace(1,300,50,dtype= int ),
                                                                                        'weights' : ["uniform", "distance"],
                                                                                        'metric' : ["cityblock", "euclidean", "minkowski"]}},
            {'type' : KMeans(), 'name' : fsel + 'KMeans', 'use' : KM, 'params' : {'n_clusters': [2],
                                                                            'max_iter' : [1000]}},
            {'type' : SVC(), 'name' : fsel + 'SVM', 'use' : SVM, 'params' : {'kernel' : ["linear", "poly", "rbf", "sigmoid"],
                                                                        'C' : np.round(np.linspace(0.001,1,15),3)}},
            {'type' : MLPClassifier(), 'name' : fsel + 'NN', 'use' : NN, 'params' : {'activation': ['logistic', 'tanh', 'relu'],
                                                                                'learning_rate_init': [0.001, 0.01, 0.1, 1, 10],
                                                                                'hidden_layer_sizes': [(), (2), (4), (2,2), (4,4)],
                                                                                'max_iter' : [1000]}}]
    # This will hold all the trained models
    trained_models = []
    # Perform hyperparameter tuning on each model and train each best one
    for model in models:
        if model['use']:
            # Check if the model was already trained
            model_path = os.path.join(os.getcwd(), 'models/' + model['name'] + '.mod')
            if not os.path.exists(model_path):
                # Perform hyperparameter tuning
                clf = GridSearchCV(model['type'], model['params'], scoring='f1', verbose=10, n_jobs=-1)
                # Fit the classifier to the data
                clf.fit(x_train, y_train.values.ravel())
                # Save the resulting best model
                dump(clf, model_path)
            else:
                # Load the model
                clf = load(model_path)
            trained_models.append(Classifier(clf, model['name']))
    return trained_models

# Show hyperparameter tuning results
def tuning_results(trained_models):
    for model in trained_models:
        # Clean up the hyperparameter tuning results 
        cv_results = model.clf.cv_results_
        cv_results = pop_substrings(cv_results, ['time', 'split'])
        print("Hyperparameter tuning results for: " + model.name)
        pd.DataFrame.from_dict(cv_results).to_csv(os.path.join(os.getcwd(), 'tuning_results/' + model.name + '_tuning_results.csv'))
        print(pd.DataFrame.from_dict(cv_results))
        plot_tuning_results(cv_results, model.name)
        
# Model validation
def model_validation(trained_models, x_test, y_test):
    # Validate each trained model against the test dataset
    for model in trained_models:
        # Make the prediction
        predict = model.clf.predict(x_test)
        # Print the classification report
        print("Classification report for: " + model.name)
        classification_metrics = classification_report(y_test, np.round(predict))
        print(classification_metrics)
        # Compute and display the confusion matrix
        plot_confusion_matrix(confusion_matrix(y_test, predict), normalize=True, classes=['No alarm', 'Alarm'])
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'plots/' + model.name + '_confusion_matrix.png'))
        plt.show()
        print(f"False Alarm Rate: {FAR(y_test, predict)}")
    # Print histogram for model comparison
    model_comparison(pd.read_csv(os.path.join(os.getcwd(), 'tuning_results/model_comparison.csv')))