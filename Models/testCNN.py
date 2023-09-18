import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report

# Custom dataset class definition
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.transform = Compose([
            Resize((32, 32)),
            ToTensor()
        ])
        self.dataset = ImageFolder(data_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

# CNN class definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization after convolution
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Normalization after convolution
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch Normalization after convolution
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)  # Batch Normalization after convolution
        self.relu4 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Batch Normalization after convolution
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Batch Normalization after convolution
        x = self.relu2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)  # Batch Normalization after convolution
        x = self.relu3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.bn4(x)  # Batch Normalization after convolution
        x = self.relu4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x

# Load the data
data_dir = "CNN/images"
dataset = CustomDataset(data_dir)

# Set seed for reproducibility
torch.manual_seed(42)

# Split the dataset into training set and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create the data loader for the test set
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
num_classes = len(dataset.dataset.classes)
model = CNN(num_classes)

# Load the weights of the trained model
model.load_state_dict(torch.load("CNN/cnn_spectrum.pth"))
model.eval()

# Test the model on the test set (use GPU if available else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

predictions_test = []
targets = []

with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        predictions_test.extend(predicted.cpu().tolist())
        targets.extend(labels.cpu().tolist())

# Test the model on the train set
predictions_train = []
targetst = []

with torch.no_grad():
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        predictions_train.extend(predicted.cpu().tolist())
        targetst.extend(labels.cpu().tolist())

# Compute the confusion matrix
cm = confusion_matrix(targets, predictions_test)

# Compute precision, recall, and f1-score
report = classification_report(targets, predictions_test)
print("\nClassification Report Test Data:")
print(report)

report = classification_report(targetst, predictions_train)
print("\nClassification Report  Train Data:")
print(report)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

class_names = dataset.dataset.classes
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Plot precision and recall per class
class_names = dataset.dataset.classes
report_dict = classification_report(targets, predictions_test, output_dict=True)

precision = []
recall = []
for i in range(num_classes):
    precision.append(report_dict[str(i)]['precision'])
    recall.append(report_dict[str(i)]['recall'])

x = np.arange(num_classes)

plt.figure(figsize=(10, 6))
plt.plot(x, precision, label='Precision')
plt.plot(x, recall, label='Recall')
plt.xlabel('Class')
plt.ylabel('Value')
plt.title('Precision and Recall per Class')
plt.xticks(x, class_names, rotation=90)
plt.legend()
plt.grid()
plt.show()
