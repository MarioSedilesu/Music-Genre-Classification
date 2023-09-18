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

# Define the parameter grid to search during the random search
param_grid = {
    'n_estimators': [100, 500, 1000],  # Number of trees
    'max_depth': [3, 5, 7],  # Maximum depth of each tree
    'learning_rate': [0.1, 0.2, 0.3]  # Learning rate
}

random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)

# Print the best parameters found
print("Best Parameters:", random_search.best_params_)

best_params = random_search.best_params_

# Create a new model with the best parameters
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)

# Calculate predictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Calculate accuracy on the test set
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy:", accuracy_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Data Accuracy:", accuracy_train)

# Save the trained model
model.save_model("TrainedModels/XGBmodel.json")
