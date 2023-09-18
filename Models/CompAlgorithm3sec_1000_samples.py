import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Load models
xgboost_model = xgb.XGBClassifier()
xgboost_model.load_model("Model_3sec_1000_samples/TrainedModels/XGBmodel.json")

random_forest_model = load("Model_3sec_1000_samples/TrainedModels/RandomForestmodel.pkl")

svm_model_poly = load("Model_3sec_1000_samples/TrainedModels/SVMPolymodel.pkl")

svm_model_rbf = load("Model_3sec_1000_samples/TrainedModels/SVMRbfmodel.pkl")

knn_model = load("Model_3sec_1000_samples/TrainedModels/KNNModel.pkl")


bayes_model = load("Model_3sec_1000_samples/TrainedModels/Bayes.pkl")

# Load dataset from CSV files
X = np.loadtxt('Model_3sec_1000_samples/Data/data.csv', delimiter='\t')
y = np.loadtxt('Model_3sec_1000_samples/Data/labels.csv', delimiter='\t')

# Combina X e y in un unico DataFrame
data = pd.DataFrame({'X': X.tolist(), 'y': y.tolist()})

# Campiona casualmente solo il 10% dei campioni per ogni classe
sampled_data = data.groupby('y', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

# Estrai X e y dai dati campionati
X_subsampled = np.array(sampled_data['X'].tolist())
y_subsampled = np.array(sampled_data['y'].tolist())

# Split the data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_subsampled, y_subsampled, test_size=0.3, random_state=42)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Make predictions for each model on the test data
xgboost_predictions_test = xgboost_model.predict(X_test)
rndforest_predictions_test = random_forest_model.predict(X_test)
svm_poly_predictions_test = svm_model_poly.predict(X_test)
svm_rtf_predictions_test = svm_model_rbf.predict(X_test)
knn_predictions_test = knn_model.predict(X_test)
bayes_predictions_test = bayes_model.predict(X_test)

# Calculate the test accuracy for each model
xgboost_accuracy_test = accuracy_score(y_test_encoded, xgboost_predictions_test)
rndforest_accuracy_test = accuracy_score(y_test_encoded, rndforest_predictions_test)
svm_poly_accuracy_test = accuracy_score(y_test_encoded, svm_poly_predictions_test)
svm_rtf_accuracy_test = accuracy_score(y_test_encoded, svm_rtf_predictions_test)
knn_accuracy_test = accuracy_score(y_test_encoded, knn_predictions_test)
bayes_accuracy_test = accuracy_score(y_test_encoded, bayes_predictions_test)

# Make predictions for each model on the train data
xgboost_predictions_train = xgboost_model.predict(X_train)
rndforest_predictions_train = random_forest_model.predict(X_train)
svm_poly_predictions_train = svm_model_poly.predict(X_train)
svm_rtf_predictions_train = svm_model_rbf.predict(X_train)
knn_predictions_train = knn_model.predict(X_train)
bayes_predictions_train = bayes_model.predict(X_train)

# Calculate the train accuracy for each model
xgboost_accuracy_train = accuracy_score(y_train_encoded, xgboost_predictions_train)
rndforest_accuracy_train = accuracy_score(y_train_encoded, rndforest_predictions_train)
svm_poly_accuracy_train = accuracy_score(y_train_encoded, svm_poly_predictions_train)
svm_rtf_accuracy_train = accuracy_score(y_train_encoded, svm_rtf_predictions_train)
knn_accuracy_train = accuracy_score(y_train_encoded, knn_predictions_train)
bayes_accuracy_train = accuracy_score(y_train_encoded, bayes_predictions_train)


# Create confusion matrices for each model
xgboost_cm = confusion_matrix(y_test_encoded, xgboost_predictions_test)
rndforest_cm = confusion_matrix(y_test_encoded, rndforest_predictions_test)
svm_poly_cm = confusion_matrix(y_test_encoded, svm_poly_predictions_test)
svm_rtf_cm = confusion_matrix(y_test_encoded, svm_rtf_predictions_test)
knn_cm = confusion_matrix(y_test_encoded, knn_predictions_test)
bayes_cm = confusion_matrix(y_test_encoded, bayes_predictions_test)

# Calculate precision, recall, and F1-score for each class
xgboost_report = classification_report(y_test_encoded, xgboost_predictions_test)
rndforest_report = classification_report(y_test_encoded, rndforest_predictions_test)
svm_poly_report = classification_report(y_test_encoded, svm_poly_predictions_test)
svm_rbf_report = classification_report(y_test_encoded, svm_rtf_predictions_test)
knn_report = classification_report(y_test_encoded, knn_predictions_test)
bayes_report = classification_report(y_test_encoded, bayes_predictions_test)

# Print the test accuracy of each model
print("XGBoost Test Accuracy:", xgboost_accuracy_test)
print("Random Test Forest Accuracy:", rndforest_accuracy_test)
print("SVM Poly Test Accuracy:", svm_poly_accuracy_test)
print("SVM Rtf Test Accuracy:", svm_rtf_accuracy_test)
print("KNN Test Accuracy:", knn_accuracy_test)
print("Bayes Test Accuracy:", bayes_accuracy_test)

# Print the test accuracy of each model
print("\nXGBoost Train Accuracy:", xgboost_accuracy_train)
print("Random Train Forest Accuracy:", rndforest_accuracy_train)
print("SVM Poly Train Accuracy:", svm_poly_accuracy_train)
print("SVM Rtf Train Accuracy:", svm_rtf_accuracy_train)
print("KNN Train Accuracy:", knn_accuracy_train)
print("Bayes Train Accuracy:", bayes_accuracy_train)

# Print precision, recall, and F1-score for each class
print("\nXGBoost Classification Report:")
print(xgboost_report)
print("Random Forest Classification Report:")
print(rndforest_report)
print("SVM Poly Classification Report:")
print(svm_poly_report)
print("SVM Rtf Classification Report:")
print(svm_rbf_report)
print("KNN Classification Report:")
print(knn_report)
print("Bayes Classification Report:")
print(bayes_report)

# Read labels
data = pd.read_csv('Data/features_3_sec.csv', delimiter=',')
class_labels = data.iloc[:, -1].unique()

# Visualize confusion matrices
plt.figure(figsize=(18, 6))

plt.subplot(231)
sns.heatmap(xgboost_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("XGBoost Confusion Matrix")

plt.subplot(232)
sns.heatmap(rndforest_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Random Forest Confusion Matrix")

plt.subplot(233)
sns.heatmap(svm_poly_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("SVM Poly Confusion Matrix")

plt.subplot(234)
sns.heatmap(svm_rtf_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("SVM Rtf Confusion Matrix")

plt.subplot(235)
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("KNN Confusion Matrix")

plt.subplot(236)
sns.heatmap(bayes_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Bayes Confusion Matrix")

plt.tight_layout()
plt.show()

# Plot accuracy, precision, and recall for each model
xgboost_report = classification_report(y_test_encoded, xgboost_predictions_test, output_dict=True)
rndforest_report = classification_report(y_test_encoded, rndforest_predictions_test, output_dict=True)
svm_poly_report = classification_report(y_test_encoded, svm_poly_predictions_test, output_dict=True)
svm_rbf_report = classification_report(y_test_encoded, svm_rtf_predictions_test, output_dict=True)
knn_report = classification_report(y_test_encoded, knn_predictions_test, output_dict=True)
bayes_report = classification_report(y_test_encoded, bayes_predictions_test, output_dict=True)

models = ['XGBoost', 'Random Forest', 'SVM Poly', 'SVM Rtf', 'KNN', 'Bayes']
accuracy = [xgboost_accuracy_test, rndforest_accuracy_test, svm_poly_accuracy_test, svm_rtf_accuracy_test, knn_accuracy_test, bayes_accuracy_test]

# Extract precision and recall values for each class
numeric_class_labels = [str(i) for i in range(10)]

xgboost_precision = [xgboost_report[label]["precision"] for label in numeric_class_labels]
xgboost_recall = [xgboost_report[label]["recall"] for label in numeric_class_labels]

rndforest_precision = [rndforest_report[label]["precision"] for label in numeric_class_labels]
rndforest_recall = [rndforest_report[label]["recall"] for label in numeric_class_labels]

svm_poly_precision = [svm_poly_report[label]["precision"] for label in numeric_class_labels]
svm_poly_recall = [svm_poly_report[label]["recall"] for label in numeric_class_labels]

svm_rbf_precision = [svm_rbf_report[label]["precision"] for label in numeric_class_labels]
svm_rbf_recall = [svm_rbf_report[label]["recall"] for label in numeric_class_labels]

knn_precision = [knn_report[label]["precision"] for label in numeric_class_labels]
knn_recall = [knn_report[label]["recall"] for label in numeric_class_labels]

bayes_precision = [bayes_report[label]["precision"] for label in numeric_class_labels]
bayes_recall = [bayes_report[label]["recall"] for label in numeric_class_labels]

# Plot accuracy, precision, and recall for each model
plt.figure(figsize=(10, 6))
plt.bar(models, accuracy)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy for each model')
plt.grid()
plt.show()

# Plot precision values
plt.figure(figsize=(12, 6))
plt.plot(class_labels, xgboost_precision, marker='o', label='XGBoost')
plt.plot(class_labels, rndforest_precision, marker='o', label='Random Forest')
plt.plot(class_labels, svm_poly_precision, marker='o', label='SVM Poly')
plt.plot(class_labels, svm_rbf_precision, marker='o', label='SVM Rbf')
plt.plot(class_labels, knn_precision, marker='o', label='KNN')
plt.plot(class_labels, bayes_precision, marker='o', label='Bayes')
plt.xlabel('Class')
plt.ylabel('Precision')
plt.title('Precision for each class')
plt.legend()
plt.grid()
plt.show()

# Plot recall values
plt.figure(figsize=(12, 6))
plt.plot(class_labels, xgboost_recall, marker='o', label='XGBoost')
plt.plot(class_labels, rndforest_recall, marker='o', label='Random Forest')
plt.plot(class_labels, svm_poly_recall, marker='o', label='SVM Poly')
plt.plot(class_labels, svm_rbf_recall, marker='o', label='SVM Rbf')
plt.plot(class_labels, knn_recall, marker='o', label='KNN')
plt.plot(class_labels, bayes_recall, marker='o', label='Bayes')
plt.xlabel('Class')
plt.ylabel('Recall')
plt.title('Recall for each class')
plt.legend()
plt.grid()
plt.show()
