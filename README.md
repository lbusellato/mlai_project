# Smoke-Detector

This repository contains the code for the final project of the Machine Learning and Artificial Intelligence course @UniVR.

The project is an implementation of a number of classical machine learning techniques for binary classification, namely:

    - Naive Bayes
    - K-Nearest Neighbors
    - Support Vector Machine
    - K-Means clustering
    - Multi-Layer Perceptron

Along these classifier models, two feature dimensionality reducton methodologies are implemented: Principal Component 
Analysis (PCA) and Linear Discriminant Analysis (LDA).

These techniques are applied to the binary classification problem of data acquired from a network of IOT sensor, in the
context of smoke and fire detection. The main goal is to obtain a comparison between the performance of different models,
as well as highlighting the role of feature selection methods on the final performance.

The report for this project is available on [Overleaf](https://www.overleaf.com/read/hyrycngnhsqc).

## Table Of Contents

- [Smoke-Detector](#Smoke-Detector)
  * [Dataset](#dataset)
  * [Setup](#setup)
  * [Usage](#usage)
  * [Authors](#authors)
  * [License](#license)

## Dataset 
The dataset picked for this project is the [Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset). The dataset is a collection of more than 60.000 readings produced from a network of IOT devices in a number of different scenarios:

    - Normal indoor
    - Normal outdoor
    - Indoor wood fire
    - Indoor gas fire
    - Outdoor wood, coal, gas grill
    - Outdoor with high umidity
    - etc.
The sensor readings constitute the dataset features, in particular:

    - Air temperature (Temperature)
    - Air humidity (Humidity)
    - Total Volatile Organic Compounds (TVOC), measured in $ppb$
    - Carbon dioxide equivalent concentration (eCO2), measured in $ppm$
    - Raw molecular hydrogen concentration (Raw H2)
    - Raw ethanol concentration (Raw Ethanol)
    - Air pressure (Pressure), measured in $hPa$.
    - Densities of particles, sizes <1.0μm  (PM1.0) and <2.5μm (PM2.5), measured in μg/m^3
    - Number of particles, sizes <0.5μm (NC0.5), <1.0μm (NC1.0) and <2.5μm (PM2.5)
    - Sample counter (CNT)
    - UTC timestamp (UTC)
    - Fire alarm, binary value (0 for no alarm, 1 for alarm) that constitutes the ground truth

## Setup

For the correct execution of the program a correctly set-up python environment.is required.
It is suggested to use an environment manager (for example Anaconda).

All required packages are listed in the requirement.txt file.

It is also possible, if using anaconda, to automatically create the correct environment by running:
```bash
foo@bar:~$ conda create --name <env> --file requirements.txt
```

For help with the usage of Anaconda see the [official documentation](https://docs.anaconda.com/)

## Usage 
A subset of the implemented methods can be actually used during execution by setting the corresponding flag in main.py:
```
BAYES = True     # Naive Bayes
KNN = True      # K Nearest Neighbors
SVM = True      # Support Vector Machine
NN = True       # Neural Network
KM = True       # K Means
```

Feature selection method can be chosen in the same way by modifying the FEATURE_SELECTOR variable at line 8.
```
FEATURE_SELECTOR = "LDA" # PCA - LDA
```

Finally, the program can be launched with:
```bash
foo@bar:~$ python3 main.py
```

After loading the dataset, the program should show the correlation matrix between the features. Then it should perform 
feature selection with the set method, and finally start the hyperparameter tuning procedure for the selected classifier 
methods. Once the classifiers are tuned, confusion matrices are produced for each one of them, as well as graphs of model 
performance with respect to hyperparameter (only for relevant classifiers). Finally, a bar plot comparing the accuracies 
and FARs of the enabled models should be produced.


## Authors
Lorenzo Busellato - lorenzo.busellato\_02@studenti.univr.it
Daniele Nicoletti - daniele.nicoletti@studenti.univr.it

## License
Smoke-Detector

Copyright © 2023 Lorenzo Busellato - Daniele Nicoletti 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
