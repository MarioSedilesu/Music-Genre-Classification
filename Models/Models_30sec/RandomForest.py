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
from sklearn.ensemble import RandomForestClassifier

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')
y = np.loadtxt('Data/labels.csv', delimiter='\t')

# Split the dataset into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of the Random Forest Classifier model
model = RandomForestClassifier()

# Define the parameter grid for random search
param_grid = {
    'n_estimators': [10, 100, 200, 300, 400, 500, 1000, 1500, 2000],
    'max_depth': [13, 14, 15, 16, 17, 18, 19, 20],
    'min_samples_split': [3, 4, 5, 6, 7, 8, 9],
    'min_samples_leaf': [3, 4, 5, 6, 7, 8, 9],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Perform random search for the best parameters using RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters found
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Train the model using the best parameters found
best_model = RandomForestClassifier(**best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test data using the trained model
y_pred = best_model.predict(X_test)

# Evaluate the model's performance using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(best_model, 'TrainedModels/RandomForestmodel.pkl')
