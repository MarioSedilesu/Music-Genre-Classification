from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import joblib

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')
y = np.loadtxt('Data/labels.csv', delimiter='\t')

# Split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM model
model = SVC()

# Define the parameter grid to search during the random search
param_grid = {
    'C': np.logspace(-3, 3, 7),  # Regularization parameter
    'kernel': ['poly'],  # SVM kernel
    'gamma': np.logspace(-3, 3, 7),  # Gamma parameter for rbf and poly kernels
    'coef0' : [0, 22, 23, 24, 25, 26, 27, 28],
    'degree': [2, 3, 4]  # Degree of polynomial kernel
}

# Perform random search for the best parameters using RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best Parameters:", random_search.best_params_)

# Get the best model
best_model = random_search.best_estimator_

# Make predictions on the test data
y_pred_test = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

# Calculate the accuracy of the model
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy:", accuracy_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Data Accuracy:", accuracy_train)

# Save the trained model
joblib.dump(best_model, 'TrainedModels/SVMPolymodel.pkl')
