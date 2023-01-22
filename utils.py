import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sb

# Remove keys from dict that contain the subkeys substrings
def pop_substrings(dict, subkeys):
    keys = dict.keys()
    pop_keys = []
    for subkey in subkeys:
        for key in keys:
            if subkey in key:
                pop_keys.append(key)
    for key in pop_keys:
        dict.pop(key, 0)
    return dict

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
             normalize=True,
             title='Confusion matrix',
             cmap=plt.cm.Blues):
    #Add Normalization Option
    '''prints pretty confusion metric with normalization option '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot hyperparameter tuning results in a very ugly way
def plot_tuning_results(grid, clf):
    data = pd.DataFrame.from_dict(grid)
    data.to_csv()
    data.drop('params', inplace=True, axis=1)
    if 'KNN' in clf:
        L = data.loc[data['param_metric'] == 'cityblock']
        l1 = L.loc[L['param_weights'] == 'uniform']
        l2 = L.loc[L['param_weights'] == 'distance']
        L = data.loc[data['param_metric'] == 'euclidean']
        l3 = L.loc[L['param_weights'] == 'uniform']
        l4 = L.loc[L['param_weights'] == 'distance']
        L = data.loc[data['param_metric'] == 'minkowski']
        l5 = L.loc[L['param_weights'] == 'uniform']
        l6 = L.loc[L['param_weights'] == 'distance']

        x =  np.array(l1['param_n_neighbors'])
        y1 = np.array(l1['mean_test_score'])
        y2 = np.array(l2['mean_test_score'])
        y3 = np.array(l3['mean_test_score'])
        y4 = np.array(l4['mean_test_score'])
        y5 = np.array(l5['mean_test_score'])
        y6 = np.array(l6['mean_test_score'])
        e1 = np.array(l1['std_test_score'])
        e2 = np.array(l2['std_test_score'])
        e3 = np.array(l3['std_test_score'])
        e4 = np.array(l4['std_test_score'])
        e5 = np.array(l5['std_test_score'])
        e6 = np.array(l6['std_test_score'])
        fig, ax = plt.subplots(1,2)
        ax[0].errorbar(x, y1, e1, linestyle='--', marker='o', label='cityblock')
        ax[0].errorbar(x, y3, e3, linestyle='--', marker='o', label='euclidean')
        ax[0].errorbar(x, y5, e5, linestyle='--', marker='o', label='minkowski')    
        ax[1].errorbar(x, y2, e2, linestyle='--', marker='o', label='cityblock')     
        ax[1].errorbar(x, y4, e4, linestyle='--', marker='o', label='euclidean')     
        ax[1].errorbar(x, y6, e6, linestyle='--', marker='o', label='minkowski')  
        ax[0].set_xlabel('K')
        ax[1].set_xlabel('K')
        ax[0].set_ylabel('Mean F1-score')    
        ax[0].grid()
        ax[1].grid()
        ax[0].legend(loc='lower left')
        ax[1].set_title('Weights: distance')
        ax[0].set_title('Weights: uniform')
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'plots/' + clf + '_hyperparameter_tuning.png'))
        plt.show()
    elif 'SVM' in clf:
        l1 = data.loc[data['param_kernel'] == 'linear']
        l2 = data.loc[data['param_kernel'] == 'poly']
        l3 = data.loc[data['param_kernel'] == 'rbf']
        l4 = data.loc[data['param_kernel'] == 'sigmoid']

        x =  np.array(l1['param_C'])
        y1 = np.array(l1['mean_test_score'])
        y2 = np.array(l2['mean_test_score'])
        y3 = np.array(l3['mean_test_score'])
        y4 = np.array(l4['mean_test_score'])
        e1 = np.array(l1['std_test_score'])
        e2 = np.array(l2['std_test_score'])
        e3 = np.array(l3['std_test_score'])
        e4 = np.array(l4['std_test_score'])
        fig, ax = plt.subplots(1,1)
        ax.errorbar(x, y1, e1, linestyle='--', marker='o', label='linear')
        ax.errorbar(x, y2, e2, linestyle='--', marker='o', label='poly')
        ax.errorbar(x, y3, e3, linestyle='--', marker='o', label='rbf')   
        ax.errorbar(x, y4, e4, linestyle='--', marker='o', label='sigmoid')   
        ax.set_xlabel('C')
        ax.set_ylabel('Mean F1-score')    
        ax.grid()
        ax.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), 'plots/' + clf + '_hyperparameter_tuning.png'))
        plt.show()

# Histogram for model comparison
def model_comparison(df):
    fig, ax  = plt.subplots(1,2)
    sb.barplot(ax=ax[0],y='Accuracy',x='Feature selection',hue='Method',data=df)
    sb.barplot(ax=ax[1],y='FAR',x='Feature selection',hue='Method',data=df)
    ax[0].grid()
    ax[0].get_legend().remove()
    ax[1].grid()
    plt.tight_layout()
    plt.show()