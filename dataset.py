import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset_path = os.path.join(os.getcwd(), 'dataset/smoke_detection_set.csv')
prepared_dataset_path = os.path.join(os.getcwd(), 'dataset/smoke_detection_set.csv')
ground_truth = ['Fire Alarm']
features = ['Temperature[C]',
            'Humidity[%]',
            'TVOC[ppb]',
            'eCO2[ppm]',
            'Raw H2',
            'Raw Ethanol',
            'Pressure[hPa]',
            'PM1.0',
            'PM2.5',
            'NC0.5',
            'NC1.0',
            'NC2.5']

# Prepare the dataset if needed, then load the train/test data
def load_data(verbose):
    # Check if the dataset was already prepared
    if not os.path.exists(prepared_dataset_path):
        # Load the dataset, replace NaN values with 0 and show info about it
        df = pd.read_csv(os.path.join(os.getcwd(), dataset_path))
        df = df.fillna(0)
        if (verbose):
            print("Dataset:")
            df.info()
        # The two classes are not balanced since there are 26884 more entries for class 
        # 1 than there are for class 0
        df1 = df[df['Fire Alarm'] == 1]
        df2 = df[df['Fire Alarm'] == 0]
        if (verbose):
            print(f"\nEntries labeled 1 (alarm): {len(df1)}")
            print(f"Entries labeled 0 (no alarm): {len(df2)}\n")
        # Sort the dataset with respect to the class labels (0 and 1)
        df.sort_values('Fire Alarm', inplace=True, ascending=False)
        # Pick the entries after index 26884, making the classes sizes equal
        df = df.iloc[26884:]
        # Reshuffle objects in the dataset 
        df = df.sample(frac=1, random_state=42)
        # Show that the classes are indeed balanced
        df1 = df[df['Fire Alarm'] == 1]
        df2 = df[df['Fire Alarm'] == 0]
        if (verbose):
            print(f"Entries labeled 1 (alarm) after class balancing: {len(df1)}")
            print(f"Entries labeled 0 (no alarm) after class balancing: {len(df2)}\n")
        # Save the prepared dataset as csv
        df.to_csv(prepared_dataset_path)
    if (verbose):
        df = pd.read_csv(prepared_dataset_path)
        print(df.head())
    # Extract the data vectors from the dataset (ignoring UTC)
    y_data = pd.read_csv(prepared_dataset_path, usecols=ground_truth)
    x_data = pd.read_csv(prepared_dataset_path, usecols=features)
    if (verbose):
        # Show the sizes of the features and labels vectors
        print("Labels : ", y_data.shape)
        print("Features : ", x_data.shape)
    # Do the train/test split on the data, keeping 80% of it as training data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    x_train = x_train.values
    x_test = x_test.values
    if (verbose):
        # Show the sizes of the resulting feature and label vectors
        print("Train labels : ", y_train.shape)
        print("Train features : ", x_train.shape)
        print("Test labels :", y_test.shape)
        print("Test features : ", x_test.shape)
    # Normalize data with min-max scaling
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
        
    return x_train, y_train, x_test, y_test

def correlation_matrix():
    # Compute the correlation between the features
    x_data = pd.read_csv(prepared_dataset_path, usecols=features)
    corr = x_data.corr()
    # Plot the correlation matrix
    plt.figure(figsize=(10,6))
    sb.heatmap(corr, annot=True).set(title='Feature correlation matrix')
    plt.tight_layout()
    plt.show()