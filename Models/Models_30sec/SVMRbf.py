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
from scipy.stats import uniform

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')
y = np.loadtxt('Data/labels.csv', delimiter='\t')

# Split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the SVM model
model = SVC()

# Define the parameter grid for random search
param_grid = {'C': [0.1, 1, 9, 10, 11, 12, 13, 100], 'gamma': [100, 10, 1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Perform random search for the best parameters using RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters found
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get the best score found
best_score = random_search.best_score_
print("Best Accuracy:", best_score)

# Get the best model
best_model = random_search.best_estimator_

# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)

# Save the trained model
joblib.dump(best_model, 'TrainedModels/SVMRbfmodel.pkl')
