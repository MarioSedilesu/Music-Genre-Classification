import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import math

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')
y = np.loadtxt('Data/labels.csv', delimiter='\t')

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the k-NN classifier
knn = KNeighborsClassifier()

# Define the parameters to explore
param_grid = {'n_neighbors': [4, 5, 6, 7, 8, 9, 10], 'weights': ['uniform', 'distance']}

# Create the GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Train the classifier with parameter search
grid_search.fit(X_train, y_train)

# Get the best classifier found
best_knn = grid_search.best_estimator_

# Predict on the test set using the best classifier
y_pred_test = best_knn.predict(X_test)
y_pred_train = best_knn.predict(X_train)

# Calculate accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy:", accuracy_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Data Accuracy:", accuracy_train)

print("Best Parameters:", best_knn)

# Save the trained model
joblib.dump(best_knn, 'TrainedModels/KNNModel.pkl')
