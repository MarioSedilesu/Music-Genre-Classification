import torch
import torch.nn as nn
import torch.optim as optim
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
data_dir = "images"
dataset = CustomDataset(data_dir)

# Set seed for reproducibility
torch.manual_seed(42)

# Split the dataset into training set and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the model
num_classes = len(dataset.dataset.classes)
model = CNN(num_classes)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
lr = 0.001  # Increased the learning rate
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)  # Added weight decay for L2 regularization

# Train the model
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_dataset)

    model.eval()
    test_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)
            correct_predictions += torch.sum(predictions == labels).item()

    test_loss /= len(test_dataset)
    test_accuracy = correct_predictions / len(test_dataset)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "cnn_spectrum.pth")

print("Best accuracy:", best_accuracy)
print("Training complete.")
