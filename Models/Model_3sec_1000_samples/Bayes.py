import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset from CSV files
X = np.loadtxt('Data/data.csv', delimiter='\t')  # Load features from a CSV file
y = np.loadtxt('Data/labels.csv', delimiter='\t')  # Load labels from a CSV file

# Combine X and y into a single DataFrame
data = pd.DataFrame({'X': X.tolist(), 'y': y.tolist()})

# Randomly sample only 10% of samples for each class
sampled_data = data.groupby('y', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

# Extract X and y from the sampled data
X_subsampled = np.array(sampled_data['X'].tolist())
y_subsampled = np.array(sampled_data['y'].tolist())

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_subsampled, y_subsampled, test_size=0.3, random_state=42)

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
