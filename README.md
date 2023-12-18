# SmallScale-ML-Projects
This repository contains small-scale projects developed to provide hands-on experience with the foundational concepts of machine learning. Each Python file in the repository corresponds to a specific concept, helping learners practice and reinforce their understanding.

# Table of Contents:
Project 1: dataframes.py
Project 2: linearRegression.py
Project 3: kmeans.py
Project 4: decisionTree.py
Project 5: randomForest.py
Project 6: cnn.py
Project 7: rnn.py
Project 8: fairness.py

## Project 1: DataFrames Functionalities
The Python script analyzes real estate data, covering tasks such as data exploration, cleaning, transformation, and statistical analysis. It includes operations like dropping null values, filtering data based on conditions, and visualizing the mean house price with respect to the number of convenience stores. The script demonstrates proficiency in pandas and data analysis techniques.

## Project 2: Linear Regression
The script performs linear regression on real estate data, assessing various metrics. It reads the dataset, renames columns, and splits the data. A loop tests the model with different random states, reporting metrics. The best-performing random state is identified. The script then highlights the most significant contributor based on coefficients and reports the intercept for the best model.

## Project 3: K-Means Clustering
This script performs KMeans clustering on customer data. After loading and exploring the dataset, it encodes the 'Gender' column using LabelEncoder. It calculates the correlation matrix and plots a heatmap. The script then identifies columns with the minimum correlation and gets the maximum correlation value. After selecting columns for clustering, it applies KMeans clustering and plots the Elbow Method to determine the optimal number of clusters. The chosen number of clusters is used to fit KMeans, and the clustered data and centroids are visualized in a scatter plot.


## Project 4: Decision Tree Classifier
The script utilizes a Decision Tree Classifier to predict diabetes outcomes based on various health-related features. It performs data loading, exploration, and preprocessing tasks. The classifier is trained, and its performance is evaluated using accuracy, precision, recall, and F1-score metrics. Additionally, the script includes a ROC curve plot and employs Stratified K-Fold Cross Validation for comprehensive model assessment. The chosen hyperparameters for the Decision Tree Classifier are evident in the code.

## Project 5: Random Forest Classifier
This script performs classification using a RandomForestClassifier on a diabetes dataset. It reads the dataset, checks for null values, and splits the data into features (X) and target values (y). A RandomForestClassifier is created, trained, and evaluated on the test set. The script reports accuracy, precision, recall, and F1 score. It also plots the ROC curve and calculates the AUC score. Finally, it uses StratifiedKFold Cross Validation to find the best k value and reports the corresponding accuracy.

## Project 6: Convolutional Neural Network
This PyTorch script implements a Convolutional Neural Network (CNN) for binary image classification. It prepares the data using torchvision, defines a CNN model, trains it on a training set, and evaluates its performance on a test set. The script prints training loss during epochs and reports metrics such as accuracy, recall, and precision for the model's evaluation.

## Project 7: Recurrent Neural Network
This PyTorch script uses a Recurrent Neural Network (RNN) to predict Bitcoin closing prices based on historical high, low, and open prices. The data is preprocessed, standardized, and split into training and testing sets. The RNN model is defined and trained, and then evaluated on the test set. The script prints training loss during epochs and calculates the R-squared score to assess the model's predictive performance.

## Project 8: Fairness in Machine Learning
The provided script demonstrates the use of the AIF360 toolkit to train and evaluate a machine learning model on the Adult dataset, considering demographic parity. The Adult dataset is preprocessed, scaled, and split into training and testing sets. Two models are trained: one without debiasing and one with debiasing. Evaluation metrics, including classification accuracy and equal opportunity difference, are calculated and printed for both models. The script utilizes TensorFlow for model training and evaluation.