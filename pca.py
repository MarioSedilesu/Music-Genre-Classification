import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

RAND_STATE = 42

# ***************** PCA Dataset 3 seconds *****************

# Load dataset
data = pd.read_csv('Models/Data/features_3_sec.csv', skiprows=1)

# Feature extraction
X = data.iloc[:, 1:-1].values  
y = data.iloc[:, -1].values

# Encoding labels in a numeric format
le = LabelEncoder()
y = le.fit_transform(y)

# Scale (Normalize) the data.
scaler = MinMaxScaler()                   
scaled_data = scaler.fit_transform(X)

#PCA
pca = PCA(random_state=RAND_STATE)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Plot the explained variance ratio 
plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained variance by number of components 3 sec')
plt.ylabel('Cumulative explained variance')
plt.xlabel('Nr. of principal components')
plt.grid()
plt.show()

explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

#Keeping only the columns with variance greater than 85%%
n_components = np.argmax(explained_variance_ratio > 0.85) + 1

new_pca = PCA(n_components=n_components)
new_pca.fit(scaled_data)
pca_data = new_pca.transform(scaled_data)

# Saving pca dataset
np.savetxt('Models/Models_3sec/Data/data.csv', pca_data, delimiter='\t')

# Saving encoded labels
np.savetxt('Models/Models_3sec/Data/labels.csv', y, delimiter='\t')

# ***************** PCA Dataset 30 seconds *****************

# Load the dataset
data = pd.read_csv('Models/Data/features_30_sec.csv', skiprows=1)

# Feature extraction
X = data.iloc[:, 1:-1].values  
y = data.iloc[:, -1].values


# Encoding lables in a numeric format
le = LabelEncoder()
y = le.fit_transform(y)

# Scale (Normalize) the data.
scaler = MinMaxScaler()                
scaled_data = scaler.fit_transform(X)  

pca = PCA(random_state=RAND_STATE)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Plot the explained variance ratio 
plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained variance by number of components 30 sec')
plt.ylabel('Cumulative explained variance')
plt.xlabel('Nr. of principal components')
plt.grid()
plt.show()

explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

#Keeping only the columns with variance greater than 85%%
n_components = np.argmax(explained_variance_ratio > 0.85) + 1

new_pca = PCA(n_components=n_components)
new_pca.fit(scaled_data)
pca_data = new_pca.transform(scaled_data)

# Save pca dataset
np.savetxt('Models/Models_30sec/Data/data.csv', pca_data, delimiter='\t')

# Save encoded labels
np.savetxt('Models/Models_30sec/Data/labels.csv', y, delimiter='\t')
