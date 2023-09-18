import torch
from torch.autograd import Variable
import torch.nn as nn
from torchviz import make_dot

# Definition of the CNN
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

# Create a dummy input tensor with adapted dimensions
batch_size = 1
num_channels = 3
input_height = 32
input_width = 32
dummy_input = Variable(torch.randn(batch_size, num_channels, input_height, input_width))

# Create the network graph
model = CNN(num_classes=10)
output = model(dummy_input)
graph = make_dot(output, params=dict(model.named_parameters()))

# Save the graph to a file
graph.render("cnn_graph")

# View the graph
graph.view("cnn_graph")
