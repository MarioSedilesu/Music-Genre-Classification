import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_subsampled, y_subsampled, test_size=0.3, random_state=42)

# Create the k-NN classifier
knn = KNeighborsClassifier()

# Define the parameters to explore
param_grid = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7], 'weights': ['uniform', 'distance']}

# Create the GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Train the classifier with parameter search
grid_search.fit(X_train, y_train)

# Get the best found classifier
best_knn = grid_search.best_estimator_

# Predict on the test set using the best classifier
y_pred_test = best_knn.predict(X_test)
y_pred_train = best_knn.predict(X_train)

# Compute accuracy
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Test Data Accuracy:", accuracy_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
print("Train Data Accuracy:", accuracy_train)

print("Best Parameters:", best_knn)

# Save the trained model
joblib.dump(best_knn, 'TrainedModels/KNNModel.pkl')
