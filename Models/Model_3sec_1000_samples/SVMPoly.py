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

# Combine X and y into a single DataFrame
data = pd.DataFrame({'X': X.tolist(), 'y': y.tolist()})

# Randomly sample only 10% of samples for each class
sampled_data = data.groupby('y', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

# Extract X and y from the sampled data
X_subsampled = np.array(sampled_data['X'].tolist())
y_subsampled = np.array(sampled_data['y'].tolist())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_subsampled, y_subsampled, test_size=0.3, random_state=42)

# Define the SVM model
model = SVC()

# Define the parameter grid to search during the random search
param_grid = {
    'C': np.logspace(-3, 3, 7),  # Regularization parameter
    'kernel': ['poly'],  # SVM kernel
    'gamma': np.logspace(-3, 3, 7),  # Gamma parameter for rbf and poly kernels
    'degree': [2, 3, 4]  # Degree of polynomial kernel
}

# Perform a random search for the best parameters using RandomizedSearchCV
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
