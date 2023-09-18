import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')
y = np.loadtxt('Data/labels.csv', delimiter='\t')

# Split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the XGBoost model
model = xgb.XGBClassifier()

# Define the parameter grid for randomized search
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200, 300, 400, 500, 1000, 1500],
    'max_depth': [3, 5, 7, 8, 9, 10, 11, 12, 13],
    'learning_rate': [0.1, 0.2, 0.3]
}

# Perform random search for the best parameters using RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)

# Get the best parameters found
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Create a new model with the best parameters
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
model.save_model("TrainedModels/XGBmodel.json")
