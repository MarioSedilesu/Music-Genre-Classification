import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')  # Load feature data
y = np.loadtxt('Data/labels.csv', delimiter='\t')  # Load label data

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set using the trained classifier
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Calculate accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy:", accuracy_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Data Accuracy:", accuracy_train)

# Save the trained model
joblib.dump(model, 'TrainedModels/Bayes.pkl')
